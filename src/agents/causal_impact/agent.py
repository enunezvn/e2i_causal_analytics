"""Causal Impact Agent - Tier 2 Hybrid Agent.

Estimates causal effects using DoWhy/EconML with natural language interpretation.
"""

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, TYPE_CHECKING

from src.agents.base import SkillsMixin
from src.agents.causal_impact.graph import create_causal_impact_graph
from src.agents.causal_impact.state import (
    CausalImpactOutput,
    CausalImpactState,
)

if TYPE_CHECKING:
    from src.agents.causal_impact.mlflow_tracker import CausalImpactMLflowTracker

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


class CausalImpactAgent(SkillsMixin):
    """Causal Impact Agent - Causal effect estimation and interpretation.

    Tier: 2 (Causal Analytics)
    Type: Hybrid (Computation + Deep Reasoning)
    SLA: 120s total (60s computation + 30s interpretation)

    Pipeline:
    1. graph_builder: Construct causal DAG (10s)
    2. estimation: Estimate ATE/CATE (30s)
    3. refutation: Robustness tests (15s)
    4. sensitivity: E-value analysis (5s)
    5. interpretation: Natural language output (30s)

    Skills Integration:
    - causal-inference/confounder-identification.md: DAG construction guidance
    - causal-inference/dowhy-workflow.md: Estimation procedures
    - pharma-commercial/brand-analytics.md: Brand-specific confounders
    """

    tier = 2
    tier_name = "causal_analytics"
    agent_type = "hybrid"  # Computation + Deep Reasoning
    agent_name = "causal_impact"  # Contract REQUIRED: BaseAgentState.agent_name
    tools = ["dowhy", "econml", "networkx"]  # Contract: AgentConfig.tools
    primary_model = "claude-sonnet-4-20250514"  # Contract: AgentConfig.primary_model
    fallback_models = ["claude-haiku-4-20250414"]  # Contract: AgentConfig.fallback_models
    memory_types: List[Literal["semantic", "episodic"]] = [
        "semantic",
        "episodic",
    ]  # Contract: AgentConfig.memory_types
    sla_seconds = 120

    def __init__(
        self,
        enable_checkpointing: bool = False,
        config: Optional[Dict[str, Any]] = None,
        enable_mlflow: bool = True,
    ):
        """Initialize Causal Impact Agent.

        Args:
            enable_checkpointing: Whether to enable state checkpointing
            config: Optional configuration overrides
            enable_mlflow: Whether to enable MLflow tracking (default: True)
        """
        self.enable_checkpointing = enable_checkpointing
        self.config = config or {}
        self.enable_mlflow = enable_mlflow

        # Initialize fallback chain (Contract: AgentConfig.fallback_models)
        self._fallback_chain = FallbackChain(self.fallback_models)

        # Memory instances (lazy initialization)
        self._semantic_memory = None
        self._episodic_memory_initialized = False

        # MLflow tracker (lazy initialization)
        self._mlflow_tracker: Optional["CausalImpactMLflowTracker"] = None

        # Create workflow graph
        self.graph = create_causal_impact_graph(enable_checkpointing)

    def _get_mlflow_tracker(self) -> Optional["CausalImpactMLflowTracker"]:
        """Get or create MLflow tracker instance (lazy initialization)."""
        if not self.enable_mlflow:
            return None

        if self._mlflow_tracker is None:
            try:
                from src.agents.causal_impact.mlflow_tracker import CausalImpactMLflowTracker

                self._mlflow_tracker = CausalImpactMLflowTracker()
            except ImportError:
                logger.warning("MLflow tracker not available")
                return None

        return self._mlflow_tracker

    async def run(self, input_data: Dict[str, Any]) -> CausalImpactOutput:
        """Execute causal impact analysis.

        Args:
            input_data: Input conforming to CausalImpactInput contract

        Returns:
            Output conforming to CausalImpactOutput contract

        Raises:
            ValueError: If required input fields are missing
        """
        start_time = time.time()

        # Clear loaded skills from previous invocation
        self.clear_loaded_skills()

        # Load relevant domain skills for this analysis
        await self._load_analysis_skills(input_data)

        # Validate required input fields per contract
        required_fields = ["query", "treatment_var", "outcome_var", "confounders", "data_source"]
        missing_fields = [f for f in required_fields if f not in input_data]
        if missing_fields:
            raise ValueError(f"Missing required field(s): {', '.join(missing_fields)}")

        # Initialize state
        initial_state = self._initialize_state(input_data)

        # Get MLflow tracker
        tracker = self._get_mlflow_tracker()

        try:
            # Run with MLflow tracking if available
            if tracker:
                async with tracker.start_analysis_run(
                    experiment_name=input_data.get("experiment_name", "default"),
                    brand=input_data.get("brand"),
                    region=input_data.get("region"),
                    treatment_var=input_data.get("treatment_var"),
                    outcome_var=input_data.get("outcome_var"),
                    query_id=initial_state.get("query_id"),
                ):
                    # Run workflow
                    final_state = await self.graph.ainvoke(initial_state)

                    # Build output
                    output = self._build_output(final_state, start_time)

                    # Log to MLflow
                    await tracker.log_analysis_result(output, final_state)

                    return output
            else:
                # Run workflow without MLflow
                final_state = await self.graph.ainvoke(initial_state)

                # Build output
                output = self._build_output(final_state, start_time)

                return output

        except Exception as e:
            # Error handling with FallbackChain (Contract: AgentConfig.fallback_models)
            latency_ms = (time.time() - start_time) * 1000
            fallback_model = self._fallback_chain.get_next()

            if fallback_model:
                logger.warning(
                    f"Primary workflow failed, attempting fallback with {fallback_model}: {e}"
                )
                try:
                    # Update state to indicate fallback
                    initial_state["fallback_used"] = True
                    initial_state["retry_count"] = initial_state.get("retry_count", 0) + 1

                    # Retry with fallback model
                    final_state = await self.graph.ainvoke(initial_state)
                    output = self._build_output(final_state, start_time)
                    output["fallback_model_used"] = fallback_model

                    # Log fallback result to MLflow
                    if tracker:
                        try:
                            await tracker.log_analysis_result(output, final_state)
                        except Exception as log_error:
                            logger.warning(f"Failed to log fallback result to MLflow: {log_error}")

                    return output

                except Exception as fallback_error:
                    logger.error(f"Fallback with {fallback_model} also failed: {fallback_error}")
                    error_output = self._build_error_output(
                        f"{str(e)} (Fallback {fallback_model} also failed: {str(fallback_error)})",
                        latency_ms,
                        input_data,
                    )
                    error_output["fallback_attempted"] = True
                    error_output["fallback_model"] = fallback_model
                    return error_output
            else:
                error_output = self._build_error_output(str(e), latency_ms, input_data)
                return error_output

    def _initialize_state(self, input_data: Dict[str, Any]) -> CausalImpactState:
        """Initialize workflow state from input.

        Args:
            input_data: Input data (must have required fields validated)

        Returns:
            Initial state conforming to contract
        """
        state: CausalImpactState = {
            # Required input fields (contract)
            "query": input_data["query"],
            "query_id": input_data.get("query_id", self._generate_query_id()),
            "treatment_var": input_data["treatment_var"],  # REQUIRED
            "outcome_var": input_data["outcome_var"],  # REQUIRED
            "confounders": input_data["confounders"],  # REQUIRED
            "data_source": input_data["data_source"],  # REQUIRED
            # Optional input fields
            "mediators": input_data.get("mediators", []),
            "effect_modifiers": input_data.get("effect_modifiers", []),
            "instruments": input_data.get("instruments", []),
            "segment_filters": input_data.get("segment_filters", {}),
            "interpretation_depth": input_data.get("interpretation_depth", "standard"),
            "user_context": input_data.get("user_context", {}),
            "parameters": input_data.get("parameters", {}),
            "time_period": input_data.get("time_period"),
            "brand": input_data.get("brand"),
            # Workflow state
            "current_phase": "graph_building",
            "status": "pending",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            # Agent identity (Contract REQUIRED: BaseAgentState.agent_name)
            "agent_name": self.agent_name,
            # Error accumulators (contract: operator.add)
            "errors": [],
            "warnings": [],
            "fallback_used": False,
            "retry_count": 0,
        }

        return state

    def _build_output(self, state: CausalImpactState, start_time: float) -> CausalImpactOutput:
        """Build output from final state.

        Args:
            state: Final workflow state
            start_time: Workflow start time

        Returns:
            Output conforming to contract
        """
        # Extract results
        interpretation = state.get("interpretation", {})
        estimation_result = state.get("estimation_result", {})
        refutation_results = state.get("refutation_results", {})
        sensitivity_analysis = state.get("sensitivity_analysis", {})
        causal_graph = state.get("causal_graph", {})

        # Calculate latencies
        computation_latency_ms = (
            state.get("graph_builder_latency_ms", 0)
            + state.get("estimation_latency_ms", 0)
            + state.get("refutation_latency_ms", 0)
            + state.get("sensitivity_latency_ms", 0)
        )

        interpretation_latency_ms = state.get("interpretation_latency_ms", 0)
        total_latency_ms = (time.time() - start_time) * 1000

        # Determine overall confidence
        refutation_confidence = refutation_results.get("confidence_adjustment", 1.0)
        sensitivity_robust = sensitivity_analysis.get("robust_to_confounding", False)
        statistical_significance = estimation_result.get("statistical_significance", False)

        if statistical_significance and sensitivity_robust:
            base_confidence = 0.9
        elif statistical_significance:
            base_confidence = 0.75
        else:
            base_confidence = 0.5

        overall_confidence = base_confidence * refutation_confidence

        # Determine refutation status
        refutation_passed = refutation_results.get("overall_robust", False)

        # Build output with contract field names
        output: CausalImpactOutput = {
            "query_id": state.get("query_id", "unknown"),
            "status": "completed",
            # Core results
            "causal_narrative": interpretation.get("narrative", "Analysis completed successfully."),
            "ate_estimate": estimation_result.get("ate"),
            "confidence_interval": (
                (
                    estimation_result.get("ate_ci_lower", 0.0),
                    estimation_result.get("ate_ci_upper", 0.0),
                )
                if "ate_ci_lower" in estimation_result
                else None
            ),
            "standard_error": estimation_result.get("standard_error"),
            "statistical_significance": estimation_result.get("statistical_significance", False),
            "p_value": estimation_result.get("p_value"),
            "effect_type": estimation_result.get("effect_type", "ate"),
            "estimation_method": estimation_result.get("method"),
            # Contract REQUIRED fields
            "confidence": overall_confidence,  # Contract field (was overall_confidence)
            "model_used": estimation_result.get("method", "unknown"),  # Contract REQUIRED
            "key_insights": interpretation.get("key_findings", []),  # Contract REQUIRED
            "assumption_warnings": self._extract_assumption_warnings(
                interpretation, estimation_result, refutation_results
            ),  # Contract REQUIRED
            "actionable_recommendations": interpretation.get(
                "recommendations", []
            ),  # Contract field (was recommendations)
            "requires_further_analysis": overall_confidence < 0.7,  # Contract REQUIRED
            "refutation_passed": refutation_passed,  # Contract REQUIRED
            "executive_summary": self._generate_executive_summary(
                interpretation, estimation_result, overall_confidence
            ),  # Contract REQUIRED
            # Rich metadata
            "mechanism_explanation": interpretation.get("mechanism_explanation"),
            "causal_graph_summary": self._summarize_causal_graph(causal_graph),
            "key_assumptions": interpretation.get("assumptions_made", []),
            "limitations": interpretation.get("limitations", []),
            # Performance metrics
            "computation_latency_ms": computation_latency_ms,
            "interpretation_latency_ms": interpretation_latency_ms,
            "total_latency_ms": total_latency_ms,
            # Robustness indicators
            "refutation_tests_passed": refutation_results.get("tests_passed"),
            "refutation_tests_total": refutation_results.get("total_tests"),
            "sensitivity_e_value": sensitivity_analysis.get("e_value"),
            # Follow-up
            "visualizations": [],
            "follow_up_suggestions": interpretation.get("recommendations", []),
            "citations": ["E2I Causal Analytics System v4.2"],
        }

        return output

    def _build_error_output(
        self, error_message: str, latency_ms: float, input_data: Dict[str, Any]
    ) -> CausalImpactOutput:
        """Build error output.

        Args:
            error_message: Error description
            latency_ms: Time elapsed before error
            input_data: Original input

        Returns:
            Error output
        """
        output: CausalImpactOutput = {
            "query_id": input_data.get("query_id", "unknown"),
            "status": "failed",
            "causal_narrative": f"Analysis failed: {error_message}",
            "statistical_significance": False,
            # Contract REQUIRED fields
            "confidence": 0.0,  # Contract field (was overall_confidence)
            "model_used": "none",  # Contract REQUIRED
            "key_insights": [],  # Contract REQUIRED
            "assumption_warnings": ["Analysis failed - unable to verify assumptions"],
            "actionable_recommendations": [
                "Review error and retry with valid input"
            ],  # Contract field (was recommendations)
            "requires_further_analysis": True,  # Contract REQUIRED
            "refutation_passed": False,  # Contract REQUIRED
            "executive_summary": f"Analysis failed: {error_message}",  # Contract REQUIRED
            # Metadata
            "key_assumptions": [],
            "limitations": ["Analysis failed before completion"],
            "computation_latency_ms": latency_ms,
            "interpretation_latency_ms": 0.0,
            "total_latency_ms": latency_ms,
            "follow_up_suggestions": [],
            "citations": [],
            "error_message": error_message,
            "partial_results": False,
        }

        return output

    def _summarize_causal_graph(self, causal_graph: Dict) -> Optional[str]:
        """Summarize causal graph structure.

        Args:
            causal_graph: CausalGraph dict

        Returns:
            Human-readable summary
        """
        if not causal_graph:
            return None

        nodes = causal_graph.get("nodes", [])
        edges = causal_graph.get("edges", [])
        treatment_nodes = causal_graph.get("treatment_nodes", [])
        outcome_nodes = causal_graph.get("outcome_nodes", [])
        adjustment_sets = causal_graph.get("adjustment_sets", [])

        summary = (
            f"Causal graph with {len(nodes)} variables and {len(edges)} relationships. "
            f"Treatment: {', '.join(treatment_nodes)}. "
            f"Outcome: {', '.join(outcome_nodes)}. "
            f"{len(adjustment_sets)} valid adjustment sets identified."
        )

        return summary

    def _extract_assumption_warnings(
        self, interpretation: Dict, estimation_result: Dict, refutation_results: Dict
    ) -> List[str]:
        """Extract assumption warnings from analysis results.

        Contract REQUIRED field: assumption_warnings

        Args:
            interpretation: Interpretation node output
            estimation_result: Estimation node output
            refutation_results: Refutation node output

        Returns:
            List of assumption warning strings
        """
        warnings = []

        # Check for low sample size
        sample_size = estimation_result.get("sample_size", 0)
        if sample_size < 100:
            warnings.append(f"Low sample size ({sample_size}) may limit reliability")

        # Check for failed refutation tests
        tests_failed = refutation_results.get("tests_failed", 0)
        if tests_failed > 0:
            warnings.append(f"{tests_failed} refutation test(s) failed - causal claim may be weak")

        # Check for weak effect size
        effect_size = estimation_result.get("effect_size", "")
        if effect_size == "small":
            warnings.append("Small effect size detected - practical significance may be limited")

        # Check assumptions from interpretation
        assumptions = interpretation.get("assumptions_made", [])
        for assumption in assumptions:
            if "unverified" in assumption.lower() or "assumed" in assumption.lower():
                warnings.append(f"Unverified assumption: {assumption}")

        # If no warnings, indicate clean status
        if not warnings:
            warnings.append("No critical assumption violations detected")

        return warnings

    def _generate_executive_summary(
        self, interpretation: Dict, estimation_result: Dict, confidence: float
    ) -> str:
        """Generate executive summary for causal impact analysis.

        Contract REQUIRED field: executive_summary (2-3 sentences)

        Args:
            interpretation: Interpretation node output
            estimation_result: Estimation node output
            confidence: Overall confidence score (0-1)

        Returns:
            Executive summary string
        """
        ate = estimation_result.get("ate")
        significance = estimation_result.get("statistical_significance", False)
        method = estimation_result.get("method", "causal inference")

        # Build summary based on results
        if ate is not None and significance:
            effect_direction = "positive" if ate > 0 else "negative"
            summary = (
                f"Analysis identified a statistically significant {effect_direction} "
                f"causal effect (ATE: {ate:.4f}) using {method}. "
                f"Overall confidence in this finding is {confidence:.0%}."
            )
        elif ate is not None:
            summary = (
                f"Analysis estimated a causal effect (ATE: {ate:.4f}) using {method}, "
                f"but the result is not statistically significant. "
                f"Confidence in this finding is {confidence:.0%}."
            )
        else:
            narrative = interpretation.get("narrative", "Analysis completed")
            summary = f"{narrative[:200]}... " if len(narrative) > 200 else narrative

        return summary

    def _generate_query_id(self) -> str:
        """Generate unique query ID.

        Returns:
            Query ID (format: q-{12 hex chars})
        """
        import secrets

        return f"q-{secrets.token_hex(6)}"

    # Helper methods for orchestrator integration

    async def classify_intent(self, query: str) -> Dict[str, Any]:
        """Helper: Classify query intent (not used by causal_impact).

        This is primarily used by orchestrator. Causal impact agent
        receives pre-classified queries.
        """
        return {
            "primary_intent": "causal_effect",
            "confidence": 0.95,
            "secondary_intents": [],
            "requires_multi_agent": False,
        }

    async def analyze(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Helper: Simplified interface for orchestrator.

        Args:
            input_data: Input data

        Returns:
            Simplified output for orchestrator synthesis
        """
        output = await self.run(input_data)

        # Return simplified format for orchestrator (contract field names)
        return {
            "narrative": output["causal_narrative"],
            "recommendations": output.get("actionable_recommendations", []),  # Contract field
            "confidence": output.get("confidence", 0.0),  # Contract field
            "key_findings": output.get("key_insights", []),  # Contract field
        }

    # ========================================================================
    # MEMORY INTEGRATION (Contract: AgentConfig.memory_types)
    # ========================================================================

    def _get_semantic_memory(self):
        """Lazy initialize semantic memory.

        Contract: memory_types includes "semantic" for graph-based entity relationships.

        Returns:
            FalkorDBSemanticMemory instance
        """
        if self._semantic_memory is None:
            from src.memory.semantic_memory import get_semantic_memory

            self._semantic_memory = get_semantic_memory()
        return self._semantic_memory

    async def query_semantic_memory(
        self, query: str, relationship_type: Optional[str] = None, max_depth: int = 2
    ) -> List[Dict[str, Any]]:
        """Query semantic memory for causal relationships.

        Contract: MemoryIntegration.query_semantic_memory

        Args:
            query: Entity ID or pattern to search
            relationship_type: Filter by relationship type (e.g., "CAUSES", "IMPACTS")
            max_depth: Maximum traversal depth

        Returns:
            List of related entities and relationships
        """
        semantic = self._get_semantic_memory()

        try:
            # Query causal chains from the entity
            chains = semantic.traverse_causal_chain(query, max_depth)

            # Filter by relationship type if specified
            if relationship_type and chains:
                chains = [
                    c
                    for c in chains
                    if any(r.get("type") == relationship_type for r in c.get("relationships", []))
                ]

            logger.debug(f"Semantic memory query for '{query}': found {len(chains)} chains")
            return chains

        except Exception as e:
            logger.warning(f"Semantic memory query failed: {e}")
            return []

    async def update_semantic_memory(self, relationships: List[Dict[str, Any]]) -> bool:
        """Update semantic memory with discovered causal relationships.

        Contract: MemoryIntegration.update_semantic_memory

        Args:
            relationships: List of relationship dicts with:
                - source_type: E2I entity type
                - source_id: Source entity ID
                - target_type: E2I entity type
                - target_id: Target entity ID
                - rel_type: Relationship type (e.g., "CAUSES")
                - properties: Optional relationship properties

        Returns:
            True if successful
        """
        from src.memory.episodic_memory import E2IEntityType

        semantic = self._get_semantic_memory()

        try:
            for rel in relationships:
                source_type = E2IEntityType(rel["source_type"])
                target_type = E2IEntityType(rel["target_type"])

                semantic.add_e2i_relationship(
                    source_type=source_type,
                    source_id=rel["source_id"],
                    target_type=target_type,
                    target_id=rel["target_id"],
                    rel_type=rel["rel_type"],
                    properties=rel.get("properties", {}),
                )

            logger.info(f"Updated semantic memory with {len(relationships)} relationships")
            return True

        except Exception as e:
            logger.error(f"Failed to update semantic memory: {e}")
            return False

    async def load_episodic_memory(
        self, query: str, top_k: int = 5, min_similarity: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Load relevant episodic memories for context.

        Contract: MemoryIntegration.load_episodic_memory

        Args:
            query: Natural language query to search
            top_k: Number of memories to retrieve
            min_similarity: Minimum similarity threshold

        Returns:
            List of relevant episodic memories
        """
        from src.memory.episodic_memory import search_episodic_by_text

        try:
            memories = await search_episodic_by_text(
                query_text=query,
                filters=None,
                limit=top_k,
                min_similarity=min_similarity,
            )

            logger.debug(f"Episodic memory query for '{query[:50]}...': found {len(memories)} memories")
            return memories

        except Exception as e:
            logger.warning(f"Episodic memory query failed: {e}")
            return []

    async def save_episodic_memory(
        self,
        event: Dict[str, Any],
        session_id: Optional[str] = None,
        cycle_id: Optional[str] = None,
    ) -> Optional[str]:
        """Save a significant analysis event to episodic memory.

        Contract: MemoryIntegration.save_episodic_memory

        Args:
            event: Event dict with:
                - event_type: Type of event (e.g., "causal_analysis")
                - description: Human-readable description
                - importance: Importance score (0-1)
                - metadata: Additional context

        Returns:
            Memory ID if successful, None otherwise
        """
        from src.memory.episodic_memory import EpisodicMemoryInput, insert_episodic_memory

        try:
            memory = EpisodicMemoryInput(
                event_type=event.get("event_type", "causal_analysis"),
                description=event.get("description", ""),
                importance=event.get("importance", 0.5),
                context_summary=event.get("context_summary"),
                action_taken=event.get("action_taken"),
                outcome=event.get("outcome"),
                emotional_valence=event.get("emotional_valence", 0.0),
            )

            memory_id = await insert_episodic_memory(
                memory=memory,
                embedding=None,  # Will be generated
                session_id=session_id,
                cycle_id=cycle_id,
            )

            logger.info(f"Saved episodic memory: {memory_id}")
            return memory_id

        except Exception as e:
            logger.error(f"Failed to save episodic memory: {e}")
            return None

    def reset_fallback_chain(self) -> None:
        """Reset fallback chain for new request.

        Call this at the start of a new analysis if reusing agent instance.
        """
        self._fallback_chain.reset()

    # ========================================================================
    # SKILLS INTEGRATION (Contract: SkillsMixin)
    # ========================================================================

    async def _load_analysis_skills(self, input_data: Dict[str, Any]) -> None:
        """Load relevant skills for the causal analysis.

        Loads domain-specific procedural knowledge based on the analysis context.
        Skills provide guidance on confounder identification, estimation methods,
        and brand-specific considerations.

        Args:
            input_data: Input data containing analysis context
        """
        try:
            # Always load core causal inference skills
            await self.load_skill("causal-inference/confounder-identification.md")
            await self.load_skill("causal-inference/dowhy-workflow.md")

            # Load brand-specific skills if brand is specified
            brand = input_data.get("brand")
            if brand:
                brand_skill = await self.load_skill("pharma-commercial/brand-analytics.md")
                if brand_skill:
                    logger.debug(f"Loaded brand analytics skill for brand: {brand}")

            # Log loaded skills
            skill_names = self.get_loaded_skill_names()
            if skill_names:
                logger.info(f"Loaded {len(skill_names)} skills for causal analysis: {skill_names}")

        except Exception as e:
            # Skills are optional - analysis should proceed without them
            logger.warning(f"Failed to load analysis skills (proceeding without): {e}")

    def get_skill_guidance(self, phase: str) -> str:
        """Get skill-based guidance for a specific analysis phase.

        Args:
            phase: Analysis phase (e.g., "graph_building", "estimation", "refutation")

        Returns:
            Relevant skill content for the phase, or empty string if unavailable.
        """
        skill_context = self.get_skill_context()
        if not skill_context:
            return ""

        # Return full context - nodes can extract relevant sections
        return skill_context
