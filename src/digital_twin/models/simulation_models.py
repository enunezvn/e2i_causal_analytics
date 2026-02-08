"""
Simulation Result Models
========================

Data models for simulation execution and results.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


class SimulationStatus(str, Enum):
    """Status of a simulation run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class SimulationRecommendation(str, Enum):
    """Recommendation based on simulation results."""

    DEPLOY = "deploy"  # Proceed to real A/B test
    SKIP = "skip"  # Do not run experiment
    REFINE = "refine"  # Refine intervention and re-simulate


class FidelityGrade(str, Enum):
    """Grade assessing twin prediction accuracy."""

    EXCELLENT = "excellent"  # Error < 10%
    GOOD = "good"  # Error 10-20%
    FAIR = "fair"  # Error 20-35%
    POOR = "poor"  # Error > 35%
    UNVALIDATED = "unvalidated"


# =============================================================================
# Intervention Configuration
# =============================================================================


class InterventionConfig(BaseModel):
    """Configuration for an intervention to simulate."""

    # Intervention type
    intervention_type: str = Field(
        description="Type of intervention (e.g., email_campaign, call_frequency)"
    )

    # Channel configuration
    channel: Optional[str] = None  # "email", "call", "in_person", "digital"
    frequency: Optional[str] = None  # "daily", "weekly", "monthly"
    duration_weeks: int = Field(default=8, ge=1, le=52)

    # Content configuration
    content_type: Optional[str] = None  # "clinical_data", "patient_stories", etc.
    personalization_level: str = "standard"  # "none", "standard", "high"

    # Targeting
    target_segment: Optional[str] = None
    target_deciles: List[int] = Field(default_factory=lambda: [1, 2, 3])
    target_specialties: List[str] = Field(default_factory=list)
    target_regions: List[str] = Field(default_factory=list)

    # Treatment intensity
    intensity_multiplier: float = Field(default=1.0, ge=0.1, le=10.0)

    # Additional parameters
    extra_params: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("target_deciles")
    @classmethod
    def validate_deciles(cls, v: List[int]) -> List[int]:
        """Ensure deciles are valid."""
        for d in v:
            if d < 1 or d > 10:
                raise ValueError(f"Invalid decile: {d}. Must be 1-10.")
        return v


class PopulationFilter(BaseModel):
    """Filters to apply when selecting twin population for simulation."""

    specialties: List[str] = Field(default_factory=list)
    deciles: List[int] = Field(default_factory=list)
    regions: List[str] = Field(default_factory=list)
    adoption_stages: List[str] = Field(default_factory=list)
    min_baseline_outcome: Optional[float] = None
    max_baseline_outcome: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "specialties": self.specialties,
            "deciles": self.deciles,
            "regions": self.regions,
            "adoption_stages": self.adoption_stages,
            "min_baseline_outcome": self.min_baseline_outcome,
            "max_baseline_outcome": self.max_baseline_outcome,
        }


# =============================================================================
# Simulation Results
# =============================================================================


class EffectHeterogeneity(BaseModel):
    """Heterogeneous effects across subgroups."""

    by_specialty: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    by_decile: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    by_region: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    by_adoption_stage: Dict[str, Dict[str, float]] = Field(default_factory=dict)

    def get_top_segments(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get top N segments by effect size."""
        segments: List[Dict[str, Any]] = []

        for dim_name, dim_data in [
            ("specialty", self.by_specialty),
            ("decile", self.by_decile),
            ("region", self.by_region),
            ("adoption_stage", self.by_adoption_stage),
        ]:
            for segment, stats in dim_data.items():
                if "ate" in stats:
                    segments.append(
                        {
                            "dimension": dim_name,
                            "segment": segment,
                            "ate": stats["ate"],
                            "n": stats.get("n", 0),
                        }
                    )

        return sorted(segments, key=lambda x: abs(float(x["ate"])), reverse=True)[:n]


class SimulationResult(BaseModel):
    """Complete results from a twin simulation."""

    # Identification
    simulation_id: UUID = Field(default_factory=uuid4)
    model_id: UUID

    # Configuration
    intervention_config: InterventionConfig
    population_filters: PopulationFilter = Field(default_factory=PopulationFilter)
    twin_count: int = Field(ge=0)  # Can be 0 for error cases (insufficient twins)

    # Effect estimates
    simulated_ate: float = Field(description="Average Treatment Effect")
    simulated_ci_lower: float = Field(description="95% CI lower bound")
    simulated_ci_upper: float = Field(description="95% CI upper bound")
    simulated_std_error: float = Field(ge=0)

    # Heterogeneity
    effect_heterogeneity: EffectHeterogeneity = Field(default_factory=EffectHeterogeneity)

    # Statistical measures
    effect_size_cohens_d: Optional[float] = None
    statistical_power: Optional[float] = None

    # Recommendation
    recommendation: SimulationRecommendation
    recommendation_rationale: str
    recommended_sample_size: Optional[int] = None
    recommended_duration_weeks: Optional[int] = None

    # Confidence and warnings
    simulation_confidence: float = Field(ge=0, le=1)
    fidelity_warning: bool = False
    fidelity_warning_reason: Optional[str] = None
    model_fidelity_score: Optional[float] = None

    # Status
    status: SimulationStatus = SimulationStatus.COMPLETED
    error_message: Optional[str] = None

    # Performance metrics
    execution_time_ms: int = Field(ge=0)
    memory_usage_mb: Optional[float] = None

    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None

    @model_validator(mode="after")
    def validate_ci_bounds(self) -> "SimulationResult":
        """Ensure CI bounds are properly ordered."""
        if self.simulated_ci_lower > self.simulated_ci_upper:
            raise ValueError("CI lower bound must be <= upper bound")
        return self

    def is_significant(self, threshold: float = 0.05) -> bool:
        """Check if effect is statistically significant (CI doesn't include 0)."""
        return self.simulated_ci_lower > 0 or self.simulated_ci_upper < 0

    def effect_direction(self) -> str:
        """Return direction of effect."""
        if self.simulated_ate > 0:
            return "positive"
        elif self.simulated_ate < 0:
            return "negative"
        return "neutral"

    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert to summary dictionary for API responses."""
        return {
            "simulation_id": str(self.simulation_id),
            "intervention_type": self.intervention_config.intervention_type,
            "twin_count": self.twin_count,
            "simulated_ate": round(self.simulated_ate, 4),
            "confidence_interval": [
                round(self.simulated_ci_lower, 4),
                round(self.simulated_ci_upper, 4),
            ],
            "recommendation": self.recommendation.value,
            "recommendation_rationale": self.recommendation_rationale,
            "recommended_sample_size": self.recommended_sample_size,
            "simulation_confidence": round(self.simulation_confidence, 3),
            "fidelity_warning": self.fidelity_warning,
            "execution_time_ms": self.execution_time_ms,
            "is_significant": self.is_significant(),
            "effect_direction": self.effect_direction(),
        }


# =============================================================================
# Fidelity Tracking
# =============================================================================


class FidelityRecord(BaseModel):
    """Record tracking simulation prediction vs. actual outcome."""

    tracking_id: UUID = Field(default_factory=uuid4)
    simulation_id: UUID

    # Predictions
    simulated_ate: float
    simulated_ci_lower: Optional[float] = None
    simulated_ci_upper: Optional[float] = None

    # Actuals (populated after real experiment)
    actual_ate: Optional[float] = None
    actual_ci_lower: Optional[float] = None
    actual_ci_upper: Optional[float] = None
    actual_sample_size: Optional[int] = None
    actual_experiment_id: Optional[UUID] = None

    # Fidelity metrics (calculated after validation)
    prediction_error: Optional[float] = None
    absolute_error: Optional[float] = None
    ci_coverage: Optional[bool] = None  # Did actual fall within predicted CI?

    # Grade
    fidelity_grade: FidelityGrade = FidelityGrade.UNVALIDATED

    # Context
    validation_notes: Optional[str] = None
    confounding_factors: List[str] = Field(default_factory=list)

    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    validated_at: Optional[datetime] = None
    validated_by: Optional[str] = None

    def calculate_fidelity(self) -> None:
        """Calculate fidelity metrics from actuals."""
        if self.actual_ate is None:
            return

        # Calculate errors
        self.prediction_error = (
            (self.simulated_ate - self.actual_ate) / self.actual_ate if self.actual_ate != 0 else 0
        )
        self.absolute_error = abs(self.simulated_ate - self.actual_ate)

        # Check CI coverage
        if self.simulated_ci_lower is not None and self.simulated_ci_upper is not None:
            self.ci_coverage = self.simulated_ci_lower <= self.actual_ate <= self.simulated_ci_upper

        # Assign grade
        abs_error = abs(self.prediction_error) if self.prediction_error else 1.0
        if abs_error < 0.10:
            self.fidelity_grade = FidelityGrade.EXCELLENT
        elif abs_error < 0.20:
            self.fidelity_grade = FidelityGrade.GOOD
        elif abs_error < 0.35:
            self.fidelity_grade = FidelityGrade.FAIR
        else:
            self.fidelity_grade = FidelityGrade.POOR

        self.validated_at = datetime.now(timezone.utc)


# =============================================================================
# Simulation Request
# =============================================================================


class SimulationRequest(BaseModel):
    """Request to run a twin simulation."""

    # Required
    intervention_type: str
    intervention_config: Dict[str, Any]
    brand: str

    # Population
    target_population: str = "hcp"  # "hcp", "patient", "territory"
    population_filters: Dict[str, Any] = Field(default_factory=dict)
    twin_count: int = Field(default=10000, ge=100, le=100000)

    # Options
    confidence_level: float = Field(default=0.95, ge=0.8, le=0.99)
    calculate_heterogeneity: bool = True

    # Metadata
    requested_by: Optional[str] = None
    experiment_design_id: Optional[UUID] = None  # Link to existing design

    def to_intervention_config(self) -> InterventionConfig:
        """Convert to InterventionConfig model."""
        return InterventionConfig(
            intervention_type=self.intervention_type,
            **self.intervention_config,
        )

    def to_population_filter(self) -> PopulationFilter:
        """Convert to PopulationFilter model."""
        return PopulationFilter(**self.population_filters)
