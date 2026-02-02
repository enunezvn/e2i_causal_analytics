"""Twin Simulation Node.

This node runs digital twin pre-screening before experiment design:
- Simulates the proposed intervention on a synthetic population
- Estimates likely treatment effect (ATE)
- Recommends DEPLOY, SKIP, or REFINE based on simulation results

If simulation recommends SKIP, the workflow exits early without
designing the full experiment, saving resources.

Phase 15 Integration:
- Uses simulate_intervention tool from digital_twin module
- Passes prior_estimate to power analysis for informed sample sizing
- Tracks simulation for later fidelity validation

Algorithm: .claude/specialists/Agent_Specialists_Tiers 1-5/experiment-designer.md
Contract: .claude/contracts/tier3-contracts.md
"""

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict

from src.agents.experiment_designer.state import ErrorDetails, ExperimentDesignState

# Import simulation tool
try:
    from src.agents.experiment_designer.tools.simulate_intervention_tool import (
        simulate_intervention,
    )

    SIMULATION_AVAILABLE = True
except ImportError:
    SIMULATION_AVAILABLE = False
    simulate_intervention = None

logger = logging.getLogger(__name__)


class TwinSimulationNode:
    """Runs digital twin pre-screening for experiment design.

    This node executes BEFORE design_reasoning to pre-filter experiments
    that are unlikely to succeed. If the simulated effect is below the
    minimum threshold, the workflow can exit early with status='skipped'.

    Workflow Integration:
        context_loader → [twin_simulation] → design_reasoning → ...

    The simulation provides:
    - Estimated ATE (Average Treatment Effect)
    - Confidence interval for the effect
    - Recommended sample size based on simulation
    - Warning if twin model fidelity is degraded

    Performance Target: <2s for 10,000 twin simulation
    """

    DEFAULT_TWIN_COUNT = 10000
    DEFAULT_CONFIDENCE_LEVEL = 0.95

    def __init__(
        self,
        min_effect_threshold: float = 0.05,
        auto_skip_on_low_effect: bool = True,
    ):
        """Initialize twin simulation node.

        Args:
            min_effect_threshold: Minimum ATE to recommend deployment (default: 0.05)
            auto_skip_on_low_effect: If True, automatically skip workflow when
                simulation recommends SKIP. If False, add warning but continue.
        """
        self.min_effect_threshold = min_effect_threshold
        self.auto_skip_on_low_effect = auto_skip_on_low_effect

    async def execute(self, state: ExperimentDesignState) -> ExperimentDesignState:
        """Execute twin simulation pre-screening.

        Args:
            state: Current agent state with business_question, intervention_type, brand

        Returns:
            Updated state with twin_simulation_result and skip_experiment flag
        """
        start_time = time.time()

        # Skip if status is failed
        if state.get("status") == "failed":
            return state

        # Check if twin simulation is enabled
        if not state.get("enable_twin_simulation", False):
            logger.info("Twin simulation disabled, skipping to design_reasoning")
            state["status"] = "reasoning"
            return state

        # Check if we have required inputs
        intervention_type = state.get("intervention_type")
        brand = state.get("brand")

        if not intervention_type or not brand:
            # Cannot run simulation without intervention details
            state["warnings"] = state.get("warnings", []) + [
                "Twin simulation skipped: missing intervention_type or brand. "
                "Proceeding with standard experiment design."
            ]
            state["status"] = "reasoning"
            return state

        # Check if simulation tool is available
        if not SIMULATION_AVAILABLE:
            state["warnings"] = state.get("warnings", []) + [
                "Twin simulation unavailable (module not installed). "
                "Proceeding with standard experiment design."
            ]
            state["status"] = "reasoning"
            return state

        try:
            # Update status
            state["status"] = "simulating_twins"

            # Extract simulation parameters from state/constraints
            constraints = state.get("constraints", {})
            simulation_params = self._extract_simulation_params(state, constraints)

            logger.info(
                f"Running twin simulation: {intervention_type} for {brand} "
                f"with {simulation_params.get('twin_count', self.DEFAULT_TWIN_COUNT)} twins"
            )

            # Run simulation
            result = simulate_intervention(
                intervention_type=intervention_type,
                brand=brand,
                target_population=simulation_params.get("target_population", "hcp"),
                channel=simulation_params.get("channel"),
                frequency=simulation_params.get("frequency"),
                duration_weeks=simulation_params.get("duration_weeks", 8),
                target_deciles=simulation_params.get("target_deciles"),
                target_specialties=simulation_params.get("target_specialties"),
                target_regions=simulation_params.get("target_regions"),
                twin_count=simulation_params.get("twin_count", self.DEFAULT_TWIN_COUNT),
                confidence_level=simulation_params.get(
                    "confidence_level", self.DEFAULT_CONFIDENCE_LEVEL
                ),
            )

            # Store simulation result in state
            state["twin_simulation_result"] = result
            state["twin_recommendation"] = result.get("recommendation", "refine")
            state["twin_simulated_ate"] = result.get("simulated_ate", 0.0)
            state["twin_recommended_sample_size"] = result.get("recommended_sample_size")
            state["twin_top_segments"] = result.get("top_segments", [])

            # Add fidelity warning if present
            if result.get("fidelity_warning"):
                state["warnings"] = state.get("warnings", []) + [
                    f"Twin model fidelity warning: {result.get('fidelity_warning_reason', 'Unknown')}"
                ]

            # Determine if we should skip
            recommendation = result.get("recommendation", "refine")

            if recommendation == "skip":
                logger.info(
                    f"Twin simulation recommends SKIP: {result.get('recommendation_rationale')}"
                )

                if self.auto_skip_on_low_effect:
                    # Early exit - don't design this experiment
                    state["skip_experiment"] = True
                    state["status"] = "skipped"
                    state["warnings"] = state.get("warnings", []) + [
                        f"Experiment skipped based on twin simulation: "
                        f"{result.get('recommendation_rationale')}"
                    ]
                else:
                    # Add warning but continue with design
                    state["skip_experiment"] = False
                    state["warnings"] = state.get("warnings", []) + [
                        f"Twin simulation recommends SKIP (proceeding anyway): "
                        f"{result.get('recommendation_rationale')}"
                    ]
                    state["status"] = "reasoning"

            elif recommendation == "refine":
                # Proceed but with caution
                state["skip_experiment"] = False
                state["warnings"] = state.get("warnings", []) + [
                    f"Twin simulation suggests refinement: {result.get('recommendation_rationale')}"
                ]
                state["status"] = "reasoning"

            else:  # "deploy"
                # Good to proceed
                state["skip_experiment"] = False
                logger.info(
                    f"Twin simulation recommends DEPLOY: "
                    f"ATE={result.get('simulated_ate'):.4f}, "
                    f"recommended_n={result.get('recommended_sample_size')}"
                )
                state["status"] = "reasoning"

            # Update latencies
            latency_ms = int((time.time() - start_time) * 1000)
            node_latencies = state.get("node_latencies_ms", {})
            node_latencies["twin_simulation"] = latency_ms
            state["node_latencies_ms"] = node_latencies

            logger.info(f"Twin simulation completed in {latency_ms}ms")

        except Exception as e:
            logger.error(f"Twin simulation failed: {e}", exc_info=True)

            error: ErrorDetails = {
                "node": "twin_simulation",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "recoverable": True,
            }
            state["errors"] = state.get("errors", []) + [error]

            # Twin simulation failure is recoverable - continue with design
            state["warnings"] = state.get("warnings", []) + [
                f"Twin simulation failed: {str(e)}. Proceeding with standard design."
            ]
            state["skip_experiment"] = False
            state["status"] = "reasoning"

        return state

    def _extract_simulation_params(
        self, state: ExperimentDesignState, constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract simulation parameters from state and constraints.

        Args:
            state: Current agent state
            constraints: Experiment constraints from input

        Returns:
            Dictionary of simulation parameters
        """
        params: Dict[str, Any] = {}

        # Target population
        if "target_population" in constraints:
            params["target_population"] = constraints["target_population"]

        # Channel and frequency
        if "channel" in constraints:
            params["channel"] = constraints["channel"]
        if "frequency" in constraints:
            params["frequency"] = constraints["frequency"]

        # Duration
        if "duration_weeks" in constraints:
            params["duration_weeks"] = constraints["duration_weeks"]
        elif "timeline_constraints" in state:
            timeline = state.get("timeline_constraints", {})
            if "duration_weeks" in timeline:
                params["duration_weeks"] = timeline["duration_weeks"]

        # Targeting filters
        if "target_deciles" in constraints:
            params["target_deciles"] = constraints["target_deciles"]
        if "target_specialties" in constraints:
            params["target_specialties"] = constraints["target_specialties"]
        if "target_regions" in constraints:
            params["target_regions"] = constraints["target_regions"]

        # Simulation settings
        if "twin_count" in constraints:
            params["twin_count"] = constraints["twin_count"]
        if "confidence_level" in constraints:
            params["confidence_level"] = constraints["confidence_level"]

        return params
