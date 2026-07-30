"""Tool Composer Agent - Tier 1 Coordination Agent.

Orchestrates multi-faceted queries by decomposing, planning, executing,
and synthesizing results from multiple agent capabilities.
"""

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from .composer import ToolComposer, ToolComposerIntegration
from .models.composition_models import CompositionResult

logger = logging.getLogger(__name__)

# Phase-appropriate canned payloads for the opt-in MARKED mock used in the
# keyless Tier 1-5 harness (#606). The composition pipeline calls the LLM three
# times (decompose -> plan -> synthesize) and parses .content as JSON with a
# DIFFERENT shape each time; a single canned blob can't satisfy all three. These
# mirror the proven shapes in tests/unit/test_agents/test_tool_composer/conftest
# (which pass the unit suite). NOT used in prod (gated on E2I_ALLOW_MOCK_LLM +
# no key); output carries mock_response_for_dev_only=True via MarkedMockChatLLM.
_MOCK_DECOMPOSITION_JSON = json.dumps(
    {
        "reasoning": "Mock decomposition for keyless harness",
        "sub_questions": [
            {
                "id": "sq_1",
                "question": "What is the causal effect of hcp_visits on discontinuation?",
                "intent": "CAUSAL",
                "entities": ["hcp_visits", "discontinuation_flag"],
                "depends_on": [],
            },
            {
                "id": "sq_2",
                "question": "How does the effect vary by prior_treatments?",
                "intent": "COMPARATIVE",
                "entities": ["prior_treatments"],
                "depends_on": ["sq_1"],
            },
        ],
    }
)
_MOCK_PLANNING_JSON = json.dumps(
    {
        "reasoning": "Mock planning for keyless harness",
        "tool_mappings": [
            {
                "sub_question_id": "sq_1",
                "tool_name": "causal_effect_estimator",
                "confidence": 0.9,
                "reasoning": "Matches causal intent",
            },
            {
                "sub_question_id": "sq_2",
                "tool_name": "causal_effect_estimator",
                "confidence": 0.85,
                "reasoning": "Second causal estimate on a different treatment",
            },
        ],
        "execution_steps": [
            {
                "step_id": "step_1",
                "sub_question_id": "sq_1",
                "tool_name": "causal_effect_estimator",
                "input_mapping": {
                    # Binary engagement treatment (median split on hcp_visits),
                    # added to context.estimation_data by the harness mapper. NOT
                    # the raw hcp_visits count: every patient has >=1 visit so the
                    # count has zero control units -> degenerate ATE (codex #606
                    # MEDIUM). step_2 uses prior_treatments (has a 0 control group).
                    "treatment": "high_hcp_engagement",
                    "outcome": "discontinuation_flag",
                    # Pull the real tier0 fixture DataFrame the harness threads via
                    # context so causal_effect_estimator runs on REAL data (#606).
                    "estimation_data": "$context.estimation_data",
                },
                "depends_on_steps": [],
            },
            {
                "step_id": "step_2",
                "sub_question_id": "sq_2",
                # Route to the REAL causal_effect_estimator on a different
                # treatment (not cate_analyzer, which returns hardcoded demo
                # segments). Two real ATEs on the threaded fixture -> genuine
                # 2/2 tool success, no fabrication (#606).
                "tool_name": "causal_effect_estimator",
                "input_mapping": {
                    "treatment": "prior_treatments",
                    "outcome": "discontinuation_flag",
                    "estimation_data": "$context.estimation_data",
                },
                "depends_on_steps": [],
            },
        ],
        "parallel_groups": [["step_1", "step_2"]],
    }
)
_MOCK_SYNTHESIS_JSON = json.dumps(
    {
        "answer": "Mock synthesis: hcp_visits shows a causal association with discontinuation.",
        "confidence": 0.85,
        "supporting_data": {"effect_size": 0.12},
        "citations": ["step_1", "step_2"],
        "caveats": ["Synthetic mock output for keyless CI (no real LLM)."],
        "failed_components": [],
        "reasoning": "Mock combined reasoning",
    }
)


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
    Aligned with ToolComposerState TypedDict for contract validation.
    """

    def __init__(
        self,
        success: bool,
        response: str,
        confidence: float,
        composition_id: Optional[str] = None,
        sub_questions_count: int = 0,
        tools_executed: int = 0,
        tools_succeeded: int = 0,
        total_duration_ms: float = 0.0,
        supporting_data: Optional[Dict[str, Any]] = None,
        citations: Optional[List[str]] = None,
        caveats: Optional[List[str]] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        status: Optional[str] = None,
    ):
        self.success = success
        self.response = response
        self.confidence = confidence
        self.composition_id = composition_id
        self.sub_questions_count = sub_questions_count
        self.tools_executed = tools_executed
        self.tools_succeeded = tools_succeeded
        self.total_duration_ms = total_duration_ms
        self.supporting_data = supporting_data or {}
        self.citations = citations or []
        self.caveats = caveats or []
        self.error = error
        self.metadata = metadata or {}
        self.timestamp = datetime.now(timezone.utc).isoformat()
        # Contract-required fields for ToolComposerState
        self.status = status or ("SUCCESS" if success else "FAILED")
        self.errors = [{"error": error}] if error else []

    def to_dict(self) -> Dict[str, Any]:
        """Convert output to dictionary."""
        return {
            "success": self.success,
            "response": self.response,
            "confidence": self.confidence,
            "composition_id": self.composition_id,
            "sub_questions_count": self.sub_questions_count,
            "tools_executed": self.tools_executed,
            "tools_succeeded": self.tools_succeeded,
            "total_duration_ms": self.total_duration_ms,
            "supporting_data": self.supporting_data,
            "citations": self.citations,
            "caveats": self.caveats,
            "error": self.error,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            # Contract-required fields
            "status": self.status,
            "errors": self.errors,
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
    primary_model = "claude-sonnet-4-6"  # Contract: AgentConfig.primary_model
    fallback_models = ["claude-haiku-4-5-20251001"]  # Contract: AgentConfig.fallback_models
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

    @staticmethod
    def _provider_key_present() -> bool:
        """Whether the configured provider's API key is available.

        Side-effect-free key probe (no client construction / no background
        threads): the factory can build a real client iff the configured
        provider's key env var is set. Used to decide between per-phase factory
        clients (key present, #1365) and the keyless MARKED mock (#606).
        """
        import os

        from src.utils.llm_factory import get_llm_provider

        key_var = "ANTHROPIC_API_KEY" if get_llm_provider() == "anthropic" else "OPENAI_API_KEY"
        return bool(os.environ.get(key_var))

    def _ensure_composer(self) -> ToolComposer:
        """Ensure composer is initialized (lazy initialization).

        Returns:
            Initialized ToolComposer instance

        Raises:
            RuntimeError: If LLM client not provided and cannot be initialized
        """
        if self._composer is None:
            if self.llm_client is None and not self._provider_key_present():
                # Keyless contexts (Tier 1-5 harness, #606): fall back to an
                # opt-in MARKED mock only when E2I_ALLOW_MOCK_LLM is set;
                # otherwise stay fail-loud (prod never gets a silent mock).
                from src.utils.mock_llm import MarkedMockChatLLM, mock_llm_allowed

                if not mock_llm_allowed():
                    raise RuntimeError(
                        "ToolComposerAgent requires an LLM client. "
                        "Provide llm_client in __init__ or set ANTHROPIC_API_KEY."
                    )
                logger.warning(
                    "tool_composer: no LLM key — using MARKED mock "
                    "(E2I_ALLOW_MOCK_LLM); output carries "
                    "mock_response_for_dev_only=True (#606)."
                )
                # Phase-aware: decompose -> plan -> synthesize each parse a
                # different JSON shape; return the right canned payload per
                # phase. Key on each node's ROLE-UNIQUE system-prompt phrase
                # ("...decomposition specialist" / "tool planning specialist"
                # / "...response synthesizer"). Loose keywords like "synth" /
                # "tool" collide because the planner embeds the tool registry,
                # which contains tools whose source_agent is
                # "prediction_synthesizer" -> "synth" hijacked the planning
                # call and fed it the synthesis JSON (#606).
                self.llm_client = MarkedMockChatLLM(
                    _MOCK_SYNTHESIS_JSON,
                    phase_responses=[
                        ("decomposition specialist", _MOCK_DECOMPOSITION_JSON),
                        ("planning specialist", _MOCK_PLANNING_JSON),
                        ("response synthesizer", _MOCK_SYNTHESIS_JSON),
                    ],
                )
            elif self.llm_client is None:
                # A provider key is present: do NOT pre-build one shared client.
                # Leaving llm_client=None lets ToolComposer build a correctly
                # SIZED client per phase (planning gets a real token budget with
                # thinking disabled — #1365), instead of the old shared
                # get_standard_llm() whose 2048/adaptive cap truncated the
                # planning JSON.
                logger.info("tool_composer: provider key present — using per-phase factory clients")

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

        # F2-core: normalize a caller-supplied DataFrame (passed as
        # input_data["data"]) into the canonical context key the executor's
        # DataFrame auto-injection reads (``estimation_data``). A passed-through
        # ``data_source`` string is also surfaced into the context for tools
        # that load by source. Only set keys when the values are actually
        # present so the executor's duck-typed gate is not tripped by None.
        data_frame = input_data.get("data")
        if data_frame is not None:
            merged_context["estimation_data"] = data_frame
        data_source = input_data.get("data_source")
        if data_source is not None:
            merged_context["data_source"] = data_source
        # #810: a KPI-aware dispatch carries the KPI outcome column so the planner
        # binds the causal outcome to the KPI (only set when present).
        kpi_outcome = input_data.get("kpi_outcome")
        if kpi_outcome is not None:
            merged_context["kpi_outcome"] = kpi_outcome

        try:
            # Ensure composer is initialized
            composer = self._ensure_composer()

            # Execute composition
            result: CompositionResult = await composer.compose(query=query, context=merged_context)

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
                tools_executed=(result.execution.tools_executed if result.execution else 0),
                tools_succeeded=(result.execution.tools_succeeded if result.execution else 0),
                total_duration_ms=result.total_duration_ms or duration_ms,
                supporting_data=(result.response.supporting_data if result.response else {}),
                citations=result.response.citations if result.response else [],
                caveats=result.response.caveats if result.response else [],
                metadata={
                    "phase_durations": result.phase_durations,
                },
                status=result.status.value.upper()
                if result.status
                else ("SUCCESS" if result.success else "FAILED"),
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
