"""Context Loader Node.

This node loads organizational learning context for experiment design:
- Historical experiments (similar past experiments)
- Organizational defaults (effect sizes, ICC values, standard confounders)
- Recent assumption violations (lessons learned)
- Domain knowledge (industry-specific constraints)

Algorithm: .claude/specialists/Agent_Specialists_Tiers 1-5/experiment-designer.md lines 86-102
Contract: .claude/contracts/tier3-contracts.md lines 82-142

Phase 4 Integration:
- Uses ExperimentKnowledgeStore to query past validation failures
- Provides warnings based on similar past failures
"""

import time
from datetime import datetime, timezone
from typing import Any, Optional

from src.agents.experiment_designer.state import ErrorDetails, ExperimentDesignState

# Phase 4: Import ExperimentKnowledgeStore for learning from past failures
try:
    from src.causal_engine.validation_outcome_store import (
        ExperimentKnowledgeStore,
        get_experiment_knowledge_store,
    )

    KNOWLEDGE_STORE_AVAILABLE = True
except ImportError:
    KNOWLEDGE_STORE_AVAILABLE = False
    ExperimentKnowledgeStore = None  # type: ignore[misc, assignment]


# ===== MOCK KNOWLEDGE STORE =====
# Fallback when ExperimentKnowledgeStore is not available
class MockKnowledgeStore:
    """Mock knowledge store for testing.

    CRITICAL: This is a temporary mock. Replace with:
        from src.repositories.knowledge_store import ExperimentKnowledgeStore
    """

    async def get_similar_experiments(
        self, business_question: str, limit: int = 5
    ) -> list[dict[str, Any]]:
        """Retrieve similar past experiments based on business question similarity."""
        return [
            {
                "experiment_id": "exp_2024_001",
                "hypothesis": "Increasing rep visit frequency improves HCP engagement",
                "design_type": "cluster_rct",
                "sample_size": 1200,
                "outcome": "engagement_score",
                "result": "positive",
                "effect_size": 0.25,
                "lessons_learned": ["Consider territory-level clustering"],
            },
            {
                "experiment_id": "exp_2024_002",
                "hypothesis": "Digital touchpoints increase prescription rates",
                "design_type": "rct",
                "sample_size": 800,
                "outcome": "trx_total",
                "result": "inconclusive",
                "effect_size": 0.12,
                "lessons_learned": ["Longer observation period needed"],
            },
        ]

    async def get_organizational_defaults(self) -> dict[str, Any]:
        """Get organization-specific default values for power analysis."""
        return {
            "effect_size": 0.25,
            "alpha": 0.05,
            "power": 0.80,
            "icc": 0.05,
            "weekly_accrual": 50,
            "standard_confounders": [
                "territory_size",
                "hcp_specialty",
                "prior_engagement",
                "competitive_intensity",
            ],
            "blocked_variables": ["region", "brand"],
        }

    async def get_recent_assumption_violations(self, limit: int = 5) -> list[dict[str, Any]]:
        """Get recent experiments where key assumptions were violated."""
        return [
            {
                "experiment_id": "exp_2023_015",
                "violation_type": "contamination",
                "description": "Control group exposed to treatment via shared resources",
                "impact": "Underestimated treatment effect by ~15%",
                "recommendation": "Use geographic separation for treatment arms",
            },
            {
                "experiment_id": "exp_2023_020",
                "violation_type": "attrition",
                "description": "25% differential attrition in treatment arm",
                "impact": "Selection bias in final estimates",
                "recommendation": "Include ITT and per-protocol analyses",
            },
        ]

    async def get_domain_knowledge(self, brand: Optional[str] = None) -> dict[str, Any]:
        """Get domain-specific knowledge and constraints."""
        return {
            "regulatory_constraints": [
                "HIPAA compliance required",
                "IRB approval for HCP surveys",
                "Pharma promotional guidelines",
            ],
            "typical_effect_sizes": {
                "rep_visits": 0.20,
                "digital_engagement": 0.15,
                "conference_attendance": 0.30,
            },
            "recommended_observation_periods": {
                "engagement_metrics": "4-8 weeks",
                "prescription_metrics": "12-16 weeks",
                "market_share": "16-24 weeks",
            },
        }


class ContextLoaderNode:
    """Loads organizational learning context for experiment design.

    This node enriches the state with historical context that helps:
    1. Design reasoning node make better design choices
    2. Power analysis use realistic effect size estimates
    3. Validity audit identify domain-specific threats

    Phase 4 Integration:
    - Queries ExperimentKnowledgeStore for past validation failures
    - Provides warnings based on similar failed validations

    Performance Target: <500ms for context loading
    """

    def __init__(
        self,
        knowledge_store: Optional[MockKnowledgeStore] = None,
        use_validation_learnings: bool = True,
    ):
        """Initialize context loader node.

        Args:
            knowledge_store: Knowledge store for retrieving context.
                            Uses ExperimentKnowledgeStore if available,
                            falls back to MockKnowledgeStore.
            use_validation_learnings: Whether to query past validation failures
        """
        self.knowledge_store = knowledge_store or MockKnowledgeStore()
        self._use_validation_learnings = use_validation_learnings
        self._experiment_knowledge_store = None

        # Phase 4: Initialize ExperimentKnowledgeStore if available
        if use_validation_learnings and KNOWLEDGE_STORE_AVAILABLE:
            try:
                self._experiment_knowledge_store = get_experiment_knowledge_store()
            except Exception:
                pass  # Fall back to mock if initialization fails

    async def execute(self, state: ExperimentDesignState) -> ExperimentDesignState:
        """Execute context loading.

        Args:
            state: Current agent state with business_question

        Returns:
            Updated state with organizational context
        """
        start_time = time.time()

        # Skip if status is failed
        if state.get("status") == "failed":
            return state

        try:
            # Update status
            state["status"] = "loading_context"

            # Load context from mock knowledge store
            similar_experiments = await self.knowledge_store.get_similar_experiments(
                state["business_question"]
            )
            organizational_defaults = await self.knowledge_store.get_organizational_defaults()
            recent_violations = await self.knowledge_store.get_recent_assumption_violations()
            domain_knowledge = await self.knowledge_store.get_domain_knowledge(state.get("brand"))

            # Phase 4: Enrich with validation failure learnings
            if self._experiment_knowledge_store is not None:
                try:
                    # Get similar experiments from validation history
                    validation_experiments = (
                        await self._experiment_knowledge_store.get_similar_experiments(
                            state["business_question"],
                            limit=3,
                        )
                    )
                    # Merge with mock experiments (validation experiments take priority)
                    if validation_experiments:
                        similar_experiments = validation_experiments + similar_experiments

                    # Get recent assumption violations from validation failures
                    validation_violations = (
                        await self._experiment_knowledge_store.get_recent_assumption_violations(
                            limit=3,
                        )
                    )
                    if validation_violations:
                        recent_violations = validation_violations + recent_violations

                except Exception as e:
                    # Non-fatal: log and continue with mock data
                    if "warnings" not in state:
                        state["warnings"] = []
                    state["warnings"].append(f"Could not load validation learnings: {str(e)}")

            # Update state with context
            state["historical_experiments"] = similar_experiments
            state["domain_knowledge"] = {
                **domain_knowledge,
                "organizational_defaults": organizational_defaults,
            }

            # Extract regulatory requirements
            state["regulatory_requirements"] = domain_knowledge.get("regulatory_constraints", [])

            # Store assumption violations for validity audit
            if "warnings" not in state:
                state["warnings"] = []

            for violation in recent_violations[
                :5
            ]:  # Increased limit to include validation learnings
                state["warnings"].append(
                    f"Past violation ({violation['violation_type']}): {violation['recommendation']}"
                )

            # Phase 4: Check for warnings specific to proposed variables
            if self._experiment_knowledge_store is not None:
                treatment_var = state.get("treatment_variable")
                outcome_var = state.get("outcome_variable")
                if treatment_var and outcome_var:
                    try:
                        design_warnings = (
                            await self._experiment_knowledge_store.should_warn_for_design(
                                treatment_variable=treatment_var,
                                outcome_variable=outcome_var,
                            )
                        )
                        state["warnings"].extend(design_warnings)
                    except Exception:
                        pass  # Non-fatal

            # Initialize execution metadata
            latency_ms = int((time.time() - start_time) * 1000)
            state["node_latencies_ms"] = {"context_loader": latency_ms}
            state["current_iteration"] = 0
            state["iteration_history"] = []

            # Update status for next node
            state["status"] = "reasoning"

        except Exception as e:
            error: ErrorDetails = {
                "node": "context_loader",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "recoverable": True,
            }
            state["errors"] = state.get("errors", []) + [error]
            # Context loading failure is recoverable - continue with empty context
            state["warnings"] = state.get("warnings", []) + [
                f"Context loading failed: {str(e)}. Proceeding with empty context."
            ]
            state["status"] = "reasoning"

        return state
