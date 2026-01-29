"""Tool Composer Agent - Tier 1 Coordination Agent.

Orchestrates multi-faceted queries by decomposing, planning, executing,
and synthesizing results from multiple agent capabilities.
"""

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from .composer import ToolComposer, ToolComposerIntegration
from .models.composition_models import CompositionResult

logger = logging.getLogger(__name__)


# ============================================================================
# FALLBACK CHAIN (Contract: AgentConfig.fallback_models)
# ============================================================================


class FallbackChain:
    """Manages fallback model progression for error handling.

    Contract: AgentConfig.fallback_models pattern from base-contract.md.
    Provides graceful degradation through alternative models.
    """

    def __init__(self, options: List[str]):
        """Initialize with list of fallback model options.

        Args:
            options: List of model names in priority order
        """
        self.options = options
        self.current_index = 0

    def get_next(self) -> Optional[str]:
        """Get next fallback option.

        Returns:
            Next model name or None if exhausted
        """
        if self.current_index < len(self.options):
            option = self.options[self.current_index]
            self.current_index += 1
            return option
        return None

    def reset(self) -> None:
        """Reset to first option."""
        self.current_index = 0

    @property
    def exhausted(self) -> bool:
        """Check if all fallbacks have been tried."""
        return self.current_index >= len(self.options)


# ============================================================================
# TOOL COMPOSER AGENT OUTPUT
# ============================================================================


class ToolComposerOutput:
    """Output from Tool Composer Agent execution.

    Contract: Standardized output format for agent responses.
    """

    def __init__(
        self,
        success: bool,
        response: str,
        confidence: float,
        composition_id: Optional[str] = None,
        sub_questions_count: int = 0,
        tools_executed: int = 0,
        total_duration_ms: float = 0.0,
        supporting_data: Optional[Dict[str, Any]] = None,
        citations: Optional[List[str]] = None,
        caveats: Optional[List[str]] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.success = success
        self.response = response
        self.confidence = confidence
        self.composition_id = composition_id
        self.sub_questions_count = sub_questions_count
        self.tools_executed = tools_executed
        self.total_duration_ms = total_duration_ms
        self.supporting_data = supporting_data or {}
        self.citations = citations or []
        self.caveats = caveats or []
        self.error = error
        self.metadata = metadata or {}
        self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert output to dictionary."""
        return {
            "success": self.success,
            "response": self.response,
            "confidence": self.confidence,
            "composition_id": self.composition_id,
            "sub_questions_count": self.sub_questions_count,
            "tools_executed": self.tools_executed,
            "total_duration_ms": self.total_duration_ms,
            "supporting_data": self.supporting_data,
            "citations": self.citations,
            "caveats": self.caveats,
            "error": self.error,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


# ============================================================================
# TOOL COMPOSER AGENT
# ============================================================================


class ToolComposerAgent:
    """Tool Composer Agent - Multi-faceted query composition.

    Tier: 1 (Coordination)
    Type: Orchestration
    SLA: 180s total (4 phases with parallelization)

    Pipeline:
    1. DECOMPOSE: Break query into atomic sub-questions (10s)
    2. PLAN: Map sub-questions to tools, create execution plan (15s)
    3. EXECUTE: Run tools in dependency order (120s)
    4. SYNTHESIZE: Combine results into coherent response (30s)

    The Tool Composer is invoked by the Orchestrator when a query is
    classified as MULTI_FACETED - requiring capabilities from multiple
    agents combined in novel ways.
    """

    tier = 1
    tier_name = "coordination"
    agent_type = "orchestration"
    agent_name = "tool_composer"  # Contract REQUIRED: BaseAgentState.agent_name
    tools = ["decomposer", "planner", "executor", "synthesizer"]
    primary_model = "claude-sonnet-4-20250514"  # Contract: AgentConfig.primary_model
    fallback_models = ["claude-haiku-4-20250414"]  # Contract: AgentConfig.fallback_models
    memory_types: List[Literal["semantic", "episodic", "procedural"]] = [
        "semantic",
        "episodic",
        "procedural",
    ]  # Contract: AgentConfig.memory_types
    sla_seconds = 180

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        enable_checkpointing: bool = False,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize Tool Composer Agent.

        Args:
            llm_client: LLM client for composition (optional, lazy init)
            enable_checkpointing: Whether to enable state checkpointing
            config: Optional configuration overrides
        """
        self.llm_client = llm_client
        self.enable_checkpointing = enable_checkpointing
        self.config = config or {}

        # Initialize fallback chain (Contract: AgentConfig.fallback_models)
        self._fallback_chain = FallbackChain(self.fallback_models)

        # Lazy initialization of composer
        self._composer: Optional[ToolComposer] = None
        self._integration: Optional[ToolComposerIntegration] = None

        logger.info("ToolComposerAgent initialized")

    def _ensure_composer(self) -> ToolComposer:
        """Ensure composer is initialized (lazy initialization).

        Returns:
            Initialized ToolComposer instance

        Raises:
            RuntimeError: If LLM client not provided and cannot be initialized
        """
        if self._composer is None:
            if self.llm_client is None:
                # Try to get default client from factory
                try:
                    from src.utils.llm_factory import get_standard_llm

                    self.llm_client = get_standard_llm()
                    logger.info("Initialized default LLM client from factory")
                except Exception as e:
                    raise RuntimeError(
                        "ToolComposerAgent requires an LLM client. "
                        "Provide llm_client in __init__ or set ANTHROPIC_API_KEY."
                    ) from e

            self._composer = ToolComposer(
                llm_client=self.llm_client,
                config=self.config,
            )
            self._integration = ToolComposerIntegration(self._composer)

        return self._composer

    async def run(self, input_data: Dict[str, Any]) -> ToolComposerOutput:
        """Execute tool composition for a multi-faceted query.

        Args:
            input_data: Input dictionary containing:
                - query (str): The multi-faceted query to process
                - context (dict, optional): Additional context
                - extracted_entities (dict, optional): Pre-extracted entities
                - user_context (dict, optional): User-specific context

        Returns:
            ToolComposerOutput with composition results

        Contract: BaseAgentState.run() - async execution entry point
        """
        start_time = time.time()

        # Extract input
        query = input_data.get("query", "")
        if not query:
            return ToolComposerOutput(
                success=False,
                response="",
                confidence=0.0,
                error="No query provided in input_data",
            )

        context = input_data.get("context", {})
        extracted_entities = input_data.get("extracted_entities", {})
        user_context = input_data.get("user_context", {})

        # Merge contexts
        merged_context = {**context, **extracted_entities, **user_context}

        try:
            # Ensure composer is initialized
            composer = self._ensure_composer()

            # Execute composition
            result: CompositionResult = await composer.compose(
                query=query, context=merged_context
            )

            # Convert to agent output
            duration_ms = (time.time() - start_time) * 1000

            return ToolComposerOutput(
                success=result.success,
                response=result.response.answer if result.response else "",
                confidence=result.response.confidence if result.response else 0.0,
                composition_id=result.composition_id,
                sub_questions_count=(
                    result.decomposition.question_count if result.decomposition else 0
                ),
                tools_executed=(
                    result.execution.tools_executed if result.execution else 0
                ),
                total_duration_ms=result.total_duration_ms or duration_ms,
                supporting_data=(
                    result.response.supporting_data if result.response else {}
                ),
                citations=result.response.citations if result.response else [],
                caveats=result.response.caveats if result.response else [],
                metadata={
                    "phase_durations": result.phase_durations,
                    "status": result.status.value if result.status else "unknown",
                },
            )

        except Exception as e:
            logger.error(f"Tool composition failed: {e}", exc_info=True)
            duration_ms = (time.time() - start_time) * 1000

            return ToolComposerOutput(
                success=False,
                response="",
                confidence=0.0,
                error=str(e),
                total_duration_ms=duration_ms,
            )

    async def handle_multi_faceted_query(
        self,
        query: str,
        extracted_entities: Dict[str, Any],
        user_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Handle a multi-faceted query from the Orchestrator.

        This is the interface used by the Orchestrator for MULTI_FACETED queries.

        Args:
            query: The classified MULTI_FACETED query
            extracted_entities: Entities extracted by the NLP layer
            user_context: User context (filters, permissions, etc.)

        Returns:
            Response dictionary in the format expected by Orchestrator
        """
        # Ensure integration is initialized
        self._ensure_composer()

        if self._integration:
            return await self._integration.handle_multi_faceted_query(
                query=query,
                extracted_entities=extracted_entities,
                user_context=user_context,
            )

        # Fallback to run method
        result = await self.run(
            {
                "query": query,
                "extracted_entities": extracted_entities,
                "user_context": user_context,
            }
        )

        return {
            "success": result.success,
            "response": result.response,
            "confidence": result.confidence,
            "supporting_data": result.supporting_data,
            "citations": result.citations,
            "caveats": result.caveats,
            "metadata": {
                "composition_id": result.composition_id,
                "sub_questions": result.sub_questions_count,
                "tools_executed": result.tools_executed,
                "total_duration_ms": result.total_duration_ms,
            },
        }

    def get_capabilities(self) -> Dict[str, Any]:
        """Return agent capabilities for registry.

        Contract: Required for agent discovery and routing.
        """
        return {
            "agent_name": self.agent_name,
            "tier": self.tier,
            "tier_name": self.tier_name,
            "agent_type": self.agent_type,
            "tools": self.tools,
            "memory_types": self.memory_types,
            "sla_seconds": self.sla_seconds,
            "description": (
                "Orchestrates multi-faceted queries by decomposing into sub-questions, "
                "mapping to tools, executing in dependency order, and synthesizing results."
            ),
            "supported_intents": [
                "MULTI_FACETED",
                "COMPARISON",
                "PREDICTION_WITH_ANALYSIS",
                "COMPLEX_QUERY",
            ],
        }
