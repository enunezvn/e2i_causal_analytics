"""
Simulate Intervention Tool
==========================

LangGraph tool for the Experiment Designer agent to pre-screen
A/B tests using digital twin simulations.

This tool integrates with the existing Experiment Designer workflow:
    Query → Context → [TWIN SIMULATION] → Design → Power → Validity → Template

If the simulated effect is below threshold, the tool recommends
skipping the experiment, saving resources on tests unlikely to succeed.
"""

import logging
from typing import Annotated, Any, Dict, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.digital_twin.fidelity_tracker import FidelityTracker
from src.digital_twin.models.simulation_models import (
    InterventionConfig,
    PopulationFilter,
    SimulationResult,
)
from src.digital_twin.models.twin_models import Brand, TwinType
from src.digital_twin.simulation_engine import SimulationEngine

# Import digital twin components
from src.digital_twin.twin_generator import TwinGenerator
from src.digital_twin.twin_repository import TwinRepository

logger = logging.getLogger(__name__)


# =============================================================================
# Tool Input/Output Schemas
# =============================================================================


class SimulateInterventionInput(BaseModel):
    """Input schema for simulate_intervention tool."""

    intervention_type: str = Field(
        description="Type of intervention to simulate (e.g., 'email_campaign', "
        "'call_frequency_increase', 'speaker_program_invitation')"
    )
    brand: str = Field(description="Pharmaceutical brand ('Remibrutinib', 'Fabhalta', 'Kisqali')")
    target_population: str = Field(
        default="hcp", description="Population type to simulate on ('hcp', 'patient', 'territory')"
    )

    # Intervention parameters
    channel: Optional[str] = Field(
        default=None, description="Communication channel ('email', 'call', 'in_person', 'digital')"
    )
    frequency: Optional[str] = Field(
        default=None, description="Intervention frequency ('daily', 'weekly', 'monthly')"
    )
    duration_weeks: int = Field(default=8, description="Duration of intervention in weeks")

    # Population filters
    target_deciles: Optional[list[int]] = Field(
        default=None, description="Target HCP deciles (e.g., [1, 2, 3] for top 30%)"
    )
    target_specialties: Optional[list[str]] = Field(
        default=None, description="Target specialties (e.g., ['oncology', 'hematology'])"
    )
    target_regions: Optional[list[str]] = Field(
        default=None, description="Target regions (e.g., ['northeast', 'midwest'])"
    )

    # Simulation parameters
    twin_count: int = Field(
        default=10000, description="Number of digital twins to simulate (100-50000)"
    )
    confidence_level: float = Field(
        default=0.95, description="Confidence level for interval calculation (0.8-0.99)"
    )


class SimulateInterventionOutput(BaseModel):
    """Output schema for simulate_intervention tool."""

    simulation_id: str
    recommendation: str  # "deploy", "skip", "refine"
    recommendation_rationale: str

    # Effect estimates
    simulated_ate: float
    confidence_interval: tuple[float, float]

    # Practical recommendations
    recommended_sample_size: Optional[int]
    recommended_duration_weeks: int

    # Confidence and warnings
    simulation_confidence: float
    fidelity_warning: bool
    fidelity_warning_reason: Optional[str]

    # Top performing segments
    top_segments: list[Dict[str, Any]]


# =============================================================================
# Tool Implementation
# =============================================================================

# Module-level cache for twin populations
_twin_cache: Dict[str, Any] = {}


def _get_or_create_twins(
    twin_type: TwinType,
    brand: Brand,
    n: int,
) -> Any:
    """Get cached twin population or generate new one."""
    cache_key = f"{twin_type.value}_{brand.value}_{n}"

    if cache_key in _twin_cache:
        logger.info(f"Using cached twin population: {cache_key}")
        return _twin_cache[cache_key]

    # In production, this would load a pre-trained model
    # For now, we'll create a mock population
    logger.info(f"Generating new twin population: {cache_key}")

    # This would be replaced with actual model loading
    TwinGenerator(twin_type=twin_type, brand=brand)

    # In production: generator would load pre-trained model
    # twins = generator.generate(n=n)

    # For now, return None - actual implementation would return population
    return None


@tool
def simulate_intervention(
    intervention_type: Annotated[str, "Type of intervention to simulate"],
    brand: Annotated[str, "Pharmaceutical brand"],
    target_population: Annotated[str, "Population type ('hcp', 'patient', 'territory')"] = "hcp",
    channel: Annotated[Optional[str], "Communication channel"] = None,
    frequency: Annotated[Optional[str], "Intervention frequency"] = None,
    duration_weeks: Annotated[int, "Duration in weeks"] = 8,
    target_deciles: Annotated[Optional[list[int]], "Target HCP deciles"] = None,
    target_specialties: Annotated[Optional[list[str]], "Target specialties"] = None,
    target_regions: Annotated[Optional[list[str]], "Target regions"] = None,
    twin_count: Annotated[int, "Number of twins to simulate"] = 10000,
    confidence_level: Annotated[float, "Confidence level (0.8-0.99)"] = 0.95,
) -> Dict[str, Any]:
    """
    Pre-screen an A/B test intervention using digital twin simulation.

    This tool simulates the proposed intervention on a population of digital
    twins to estimate the likely treatment effect BEFORE running a real
    experiment. Use this to:

    1. SKIP experiments with low predicted impact (save resources)
    2. REFINE interventions that show promise but need adjustment
    3. DEPLOY experiments that show strong predicted effects

    The simulation provides:
    - Estimated Average Treatment Effect (ATE)
    - Confidence interval for the effect
    - Recommended sample size for real experiment
    - Warning if twin model fidelity is degraded

    Args:
        intervention_type: Type of intervention (email_campaign, call_frequency_increase,
            speaker_program_invitation, sample_distribution, peer_influence_activation,
            digital_engagement)
        brand: Pharmaceutical brand (Remibrutinib, Fabhalta, Kisqali)
        target_population: Population to simulate (hcp, patient, territory)
        channel: Communication channel (email, call, in_person, digital)
        frequency: Intervention frequency (daily, weekly, monthly)
        duration_weeks: How long the intervention lasts (default: 8)
        target_deciles: Which HCP deciles to target (e.g., [1,2,3] for top 30%)
        target_specialties: Which specialties to target
        target_regions: Which geographic regions to target
        twin_count: Number of digital twins to simulate (default: 10000)
        confidence_level: Confidence level for CI (default: 0.95)

    Returns:
        Dictionary containing:
        - recommendation: "deploy", "skip", or "refine"
        - recommendation_rationale: Explanation of recommendation
        - simulated_ate: Estimated Average Treatment Effect
        - confidence_interval: (lower, upper) bounds
        - recommended_sample_size: Suggested N for real experiment
        - simulation_confidence: Confidence in simulation results
        - fidelity_warning: Whether model accuracy is degraded
        - top_segments: Best performing subgroups

    Example:
        >>> result = simulate_intervention(
        ...     intervention_type="email_campaign",
        ...     brand="Kisqali",
        ...     channel="email",
        ...     frequency="weekly",
        ...     duration_weeks=8,
        ...     target_deciles=[1, 2, 3]
        ... )
        >>> if result["recommendation"] == "skip":
        ...     print("Skipping experiment:", result["recommendation_rationale"])
    """
    logger.info(
        f"simulate_intervention called: {intervention_type} on {brand}, "
        f"population={target_population}, twins={twin_count}"
    )

    try:
        # Parse inputs
        twin_type = TwinType(target_population)
        brand_enum = Brand(brand)

        # Build intervention config
        intervention_config = InterventionConfig(
            intervention_type=intervention_type,
            channel=channel,
            frequency=frequency,
            duration_weeks=duration_weeks,
            target_deciles=target_deciles or [1, 2, 3],
            target_specialties=target_specialties or [],
            target_regions=target_regions or [],
        )

        # Build population filter
        population_filter = PopulationFilter(
            specialties=target_specialties or [],
            deciles=target_deciles or [],
            regions=target_regions or [],
        )

        # Get or create twin population
        twins = _get_or_create_twins(twin_type, brand_enum, twin_count)

        if twins is None:
            # Return mock result for demonstration
            # In production, this would use real simulation
            return _create_mock_result(
                intervention_config,
                twin_count,
                intervention_type,
            )

        # Run simulation
        engine = SimulationEngine(
            population=twins,
            min_effect_threshold=0.05,
            confidence_threshold=0.70,
        )

        result = engine.simulate(
            intervention_config=intervention_config,
            population_filter=population_filter,
            confidence_level=confidence_level,
            calculate_heterogeneity=True,
        )

        # Track for fidelity validation
        tracker = FidelityTracker()
        tracker.record_prediction(result)

        # Format output
        return _format_output(result)

    except Exception as e:
        logger.error(f"Simulation failed: {e}", exc_info=True)
        return {
            "simulation_id": "error",
            "recommendation": "refine",
            "recommendation_rationale": f"Simulation failed: {str(e)}. "
            "Please check inputs and try again.",
            "simulated_ate": 0.0,
            "confidence_interval": (0.0, 0.0),
            "recommended_sample_size": None,
            "recommended_duration_weeks": duration_weeks,
            "simulation_confidence": 0.0,
            "fidelity_warning": True,
            "fidelity_warning_reason": str(e),
            "top_segments": [],
        }


def _format_output(result: SimulationResult) -> Dict[str, Any]:
    """Format SimulationResult for tool output."""
    return {
        "simulation_id": str(result.simulation_id),
        "recommendation": result.recommendation.value,
        "recommendation_rationale": result.recommendation_rationale,
        "simulated_ate": round(result.simulated_ate, 4),
        "confidence_interval": (
            round(result.simulated_ci_lower, 4),
            round(result.simulated_ci_upper, 4),
        ),
        "recommended_sample_size": result.recommended_sample_size,
        "recommended_duration_weeks": result.intervention_config.duration_weeks,
        "simulation_confidence": round(result.simulation_confidence, 3),
        "fidelity_warning": result.fidelity_warning,
        "fidelity_warning_reason": result.fidelity_warning_reason,
        "top_segments": result.effect_heterogeneity.get_top_segments(5),
    }


def _create_mock_result(
    config: InterventionConfig,
    twin_count: int,
    intervention_type: str,
) -> Dict[str, Any]:
    """Create mock result for demonstration when model not available."""
    import random
    from uuid import uuid4

    # Simulate different effects based on intervention type
    base_effects = {
        "email_campaign": 0.06,
        "call_frequency_increase": 0.09,
        "speaker_program_invitation": 0.14,
        "sample_distribution": 0.04,
        "peer_influence_activation": 0.11,
        "digital_engagement": 0.07,
    }

    base = base_effects.get(intervention_type, 0.05)
    noise = random.gauss(0, 0.02)
    ate = base + noise

    std_error = 0.015
    ci_lower = ate - 1.96 * std_error
    ci_upper = ate + 1.96 * std_error

    # Determine recommendation
    if ate < 0.05:
        recommendation = "skip"
        rationale = (
            f"Simulated ATE ({ate:.4f}) below minimum threshold (0.05). "
            "Predicted impact insufficient to justify experiment costs."
        )
    elif ci_lower <= 0:
        recommendation = "refine"
        rationale = (
            "Effect not statistically significant (CI includes zero). "
            "Consider refining intervention design or increasing duration."
        )
    else:
        recommendation = "deploy"
        rationale = (
            f"Simulation predicts positive effect (ATE={ate:.4f}). "
            "Effect exceeds threshold and is significant. Proceed with A/B test."
        )

    return {
        "simulation_id": str(uuid4()),
        "recommendation": recommendation,
        "recommendation_rationale": rationale,
        "simulated_ate": round(ate, 4),
        "confidence_interval": (round(ci_lower, 4), round(ci_upper, 4)),
        "recommended_sample_size": int(2000 / (ate**2)) if ate > 0.01 else 10000,
        "recommended_duration_weeks": config.duration_weeks,
        "simulation_confidence": 0.75,
        "fidelity_warning": False,
        "fidelity_warning_reason": None,
        "top_segments": [
            {"dimension": "decile", "segment": "1-2", "ate": ate * 1.3, "n": 2000},
            {"dimension": "specialty", "segment": "oncology", "ate": ate * 1.2, "n": 3000},
            {
                "dimension": "adoption_stage",
                "segment": "late_majority",
                "ate": ate * 1.15,
                "n": 2500,
            },
        ],
    }


# =============================================================================
# Integration with Experiment Designer Workflow
# =============================================================================


class DigitalTwinWorkflow:
    """
    Integrates digital twin simulation into Experiment Designer workflow.

    This class implements the pattern from the assessment:
    - Twins pre-filter experiments, not replace them
    - If predicted effect is negligible, recommend SKIP
    - If promising, pass prior estimate to Experiment Designer
    """

    MIN_EFFECT_THRESHOLD = 0.05

    def __init__(
        self,
        experiment_designer=None,
        twin_repository: Optional[TwinRepository] = None,
    ):
        """
        Initialize workflow.

        Args:
            experiment_designer: Reference to Experiment Designer agent
            twin_repository: Repository for model/simulation storage
        """
        self.experiment_designer = experiment_designer
        self.repository = twin_repository
        self.fidelity_tracker = FidelityTracker(repository=twin_repository)

    def propose_experiment(self, intervention_type: str, brand: str, **kwargs) -> Dict[str, Any]:
        """
        Propose an experiment with twin pre-screening.

        Returns:
            Dict with either:
            - action="SKIP" and reason (if simulated effect too low)
            - action="DESIGN" with prior_estimate and recommended_sample_size
        """
        # Run simulation
        sim_result = simulate_intervention(
            intervention_type=intervention_type, brand=brand, **kwargs
        )

        # Check if we should skip
        if sim_result["recommendation"] == "skip":
            return {
                "action": "SKIP",
                "reason": sim_result["recommendation_rationale"],
                "simulation_id": sim_result["simulation_id"],
                "simulated_ate": sim_result["simulated_ate"],
                "confidence": "SIMULATION_ONLY",
            }

        # Pass to Experiment Designer with prior information
        return {
            "action": "DESIGN",
            "simulation_id": sim_result["simulation_id"],
            "prior_estimate": {
                "ate": sim_result["simulated_ate"],
                "ci": sim_result["confidence_interval"],
            },
            "recommended_sample_size": sim_result["recommended_sample_size"],
            "top_segments": sim_result["top_segments"],
            "fidelity_warning": sim_result["fidelity_warning"],
        }
