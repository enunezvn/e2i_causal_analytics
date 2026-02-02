"""
E2I Digital Twin Pre-Screening API
===================================

FastAPI endpoints for Digital Twin simulation, fidelity tracking, and model management.

Phase 15: Digital Twin Pre-Screening for A/B Tests

Endpoints:
- POST /digital-twin/simulate: Run twin simulation for an intervention
- GET /digital-twin/simulations: List simulation results
- GET /digital-twin/simulations/{id}: Get simulation details
- POST /digital-twin/validate: Validate simulation against actual experiment results
- GET /digital-twin/models: List trained twin generator models
- GET /digital-twin/models/{id}: Get model details
- GET /digital-twin/models/{id}/fidelity: Get fidelity history for a model

Integration Points:
- TwinGenerator: ML-based twin generation
- SimulationEngine: Intervention effect simulation
- FidelityTracker: Prediction accuracy tracking
- TwinRepository: Persistence layer

Author: E2I Causal Analytics Team
Version: 4.2.0
"""

import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from src.api.dependencies.auth import require_operator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/digital-twin", tags=["Digital Twin"])


# =============================================================================
# ENUMS
# =============================================================================


class TwinTypeEnum(str, Enum):
    """Types of digital twins."""

    HCP = "hcp"
    PATIENT = "patient"
    TERRITORY = "territory"


class BrandEnum(str, Enum):
    """Pharmaceutical brands."""

    REMIBRUTINIB = "Remibrutinib"
    FABHALTA = "Fabhalta"
    KISQALI = "Kisqali"


class SimulationStatusEnum(str, Enum):
    """Simulation status values."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class RecommendationEnum(str, Enum):
    """Simulation recommendations."""

    DEPLOY = "deploy"
    SKIP = "skip"
    REFINE = "refine"


class FidelityGradeEnum(str, Enum):
    """Fidelity grade values."""

    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    UNVALIDATED = "unvalidated"


# =============================================================================
# REQUEST MODELS
# =============================================================================


class InterventionConfigRequest(BaseModel):
    """Configuration for an intervention to simulate."""

    intervention_type: str = Field(
        ..., description="Type of intervention (email_campaign, call_frequency_increase, etc.)"
    )
    channel: Optional[str] = Field(None, description="Channel: email, call, in_person, digital")
    frequency: Optional[str] = Field(None, description="Frequency: daily, weekly, monthly")
    duration_weeks: int = Field(default=8, ge=1, le=52, description="Duration in weeks")
    content_type: Optional[str] = Field(
        None, description="Content type: clinical_data, patient_stories, etc."
    )
    personalization_level: str = Field(default="standard", description="none, standard, high")
    target_segment: Optional[str] = Field(None, description="Target segment identifier")
    target_deciles: List[int] = Field(default=[1, 2, 3], description="Target deciles (1-10)")
    target_specialties: List[str] = Field(default=[], description="Target specialty list")
    target_regions: List[str] = Field(default=[], description="Target region list")
    intensity_multiplier: float = Field(
        default=1.0, ge=0.1, le=10.0, description="Treatment intensity"
    )
    extra_params: Dict[str, Any] = Field(default={}, description="Additional parameters")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "intervention_type": "email_campaign",
                "channel": "email",
                "frequency": "weekly",
                "duration_weeks": 8,
                "personalization_level": "high",
                "target_deciles": [1, 2, 3],
            }
        }
    )


class PopulationFilterRequest(BaseModel):
    """Filters for selecting twin population."""

    specialties: List[str] = Field(default=[], description="Filter by specialties")
    deciles: List[int] = Field(default=[], description="Filter by deciles (1-10)")
    regions: List[str] = Field(default=[], description="Filter by regions")
    adoption_stages: List[str] = Field(default=[], description="Filter by adoption stages")
    min_baseline_outcome: Optional[float] = Field(None, description="Minimum baseline outcome")
    max_baseline_outcome: Optional[float] = Field(None, description="Maximum baseline outcome")


class SimulateRequest(BaseModel):
    """Request to run a twin simulation."""

    intervention: InterventionConfigRequest
    brand: BrandEnum
    twin_type: TwinTypeEnum = Field(default=TwinTypeEnum.HCP)
    population_filters: Optional[PopulationFilterRequest] = None
    twin_count: int = Field(
        default=10000, ge=100, le=100000, description="Number of twins to simulate"
    )
    confidence_level: float = Field(
        default=0.95, ge=0.8, le=0.99, description="Confidence level for CI"
    )
    calculate_heterogeneity: bool = Field(
        default=True, description="Calculate heterogeneous effects"
    )
    model_id: Optional[str] = Field(None, description="Specific model ID to use")
    experiment_design_id: Optional[str] = Field(None, description="Link to experiment design")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "intervention": {
                    "intervention_type": "email_campaign",
                    "channel": "email",
                    "duration_weeks": 8,
                },
                "brand": "Remibrutinib",
                "twin_type": "hcp",
                "twin_count": 10000,
                "population_filters": {"deciles": [1, 2, 3]},
            }
        }
    )


class ValidateFidelityRequest(BaseModel):
    """Request to validate simulation against actual results."""

    simulation_id: str = Field(..., description="Simulation ID to validate")
    experiment_id: str = Field(..., description="Actual experiment ID")
    actual_ate: float = Field(..., description="Actual Average Treatment Effect")
    actual_ci_lower: Optional[float] = Field(None, description="Actual CI lower bound")
    actual_ci_upper: Optional[float] = Field(None, description="Actual CI upper bound")
    actual_sample_size: Optional[int] = Field(None, description="Actual sample size")
    validation_notes: Optional[str] = Field(None, description="Notes on validation")
    confounding_factors: List[str] = Field(default=[], description="Known confounding factors")
    validated_by: Optional[str] = Field(None, description="Validator identifier")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "simulation_id": "550e8400-e29b-41d4-a716-446655440000",
                "experiment_id": "660e8400-e29b-41d4-a716-446655440000",
                "actual_ate": 0.072,
                "actual_ci_lower": 0.045,
                "actual_ci_upper": 0.099,
                "actual_sample_size": 5000,
            }
        }
    )


# =============================================================================
# RESPONSE MODELS
# =============================================================================


class EffectHeterogeneityResponse(BaseModel):
    """Heterogeneous effects across subgroups."""

    by_specialty: Dict[str, Dict[str, float]] = Field(default={})
    by_decile: Dict[str, Dict[str, float]] = Field(default={})
    by_region: Dict[str, Dict[str, float]] = Field(default={})
    by_adoption_stage: Dict[str, Dict[str, float]] = Field(default={})
    top_segments: List[Dict[str, Any]] = Field(default=[])


class SimulationResponse(BaseModel):
    """Response from a simulation run."""

    simulation_id: str
    model_id: str
    intervention_type: str
    brand: str
    twin_type: str
    twin_count: int
    simulated_ate: float
    simulated_ci_lower: float
    simulated_ci_upper: float
    simulated_std_error: float
    effect_size_cohens_d: Optional[float] = None
    statistical_power: Optional[float] = None
    recommendation: RecommendationEnum
    recommendation_rationale: str
    recommended_sample_size: Optional[int] = None
    recommended_duration_weeks: Optional[int] = None
    simulation_confidence: float
    fidelity_warning: bool
    fidelity_warning_reason: Optional[str] = None
    model_fidelity_score: Optional[float] = None
    status: SimulationStatusEnum
    error_message: Optional[str] = None
    execution_time_ms: int
    is_significant: bool
    effect_direction: str
    created_at: datetime


class SimulationDetailResponse(SimulationResponse):
    """Detailed simulation response including heterogeneity."""

    population_filters: Dict[str, Any]
    effect_heterogeneity: EffectHeterogeneityResponse
    intervention_config: Dict[str, Any]
    completed_at: Optional[datetime] = None


class SimulationListItem(BaseModel):
    """Summary item for simulation list."""

    simulation_id: str
    intervention_type: str
    brand: str
    twin_type: str
    twin_count: int
    simulated_ate: float
    recommendation: RecommendationEnum
    status: SimulationStatusEnum
    created_at: datetime


class SimulationListResponse(BaseModel):
    """Response for listing simulations."""

    total_count: int
    simulations: List[SimulationListItem]
    page: int
    page_size: int


class FidelityRecordResponse(BaseModel):
    """Fidelity validation record."""

    tracking_id: str
    simulation_id: str
    experiment_id: Optional[str] = None
    simulated_ate: float
    simulated_ci_lower: Optional[float] = None
    simulated_ci_upper: Optional[float] = None
    actual_ate: Optional[float] = None
    actual_ci_lower: Optional[float] = None
    actual_ci_upper: Optional[float] = None
    actual_sample_size: Optional[int] = None
    prediction_error: Optional[float] = None
    absolute_error: Optional[float] = None
    ci_coverage: Optional[bool] = None
    fidelity_grade: FidelityGradeEnum
    validation_notes: Optional[str] = None
    confounding_factors: List[str] = []
    created_at: datetime
    validated_at: Optional[datetime] = None
    validated_by: Optional[str] = None


class TwinModelSummary(BaseModel):
    """Summary of a twin generator model."""

    model_id: str
    model_name: str
    twin_type: str
    brand: str
    algorithm: str
    r2_score: Optional[float] = None
    rmse: Optional[float] = None
    training_samples: int
    is_active: bool
    created_at: datetime


class TwinModelDetailResponse(TwinModelSummary):
    """Detailed twin model information."""

    model_description: Optional[str] = None
    feature_columns: List[str]
    target_column: str
    cv_mean: Optional[float] = None
    cv_std: Optional[float] = None
    feature_importances: Dict[str, float]
    top_features: List[str]
    training_duration_seconds: float
    config: Dict[str, Any]


class ModelListResponse(BaseModel):
    """Response for listing models."""

    total_count: int
    models: List[TwinModelSummary]


class FidelityHistoryResponse(BaseModel):
    """Fidelity history for a model."""

    model_id: str
    total_validations: int
    average_fidelity_score: Optional[float] = None
    grade_distribution: Dict[str, int]
    records: List[FidelityRecordResponse]


class FidelityReportResponse(BaseModel):
    """Aggregated fidelity report for a model."""

    model_id: str
    total_validations: int
    average_fidelity_score: float
    coverage_rate: float
    grade_distribution: Dict[str, int]
    trend: str
    is_degrading: bool
    degradation_rate: Optional[float] = None
    recommendation: str
    generated_at: datetime


# =============================================================================
# HEALTH ENDPOINTS
# =============================================================================


class DigitalTwinHealthResponse(BaseModel):
    """Health status for Digital Twin service."""

    status: str = Field(..., description="Service health status")
    service: str = Field(default="digital-twin", description="Service name")
    models_available: int = Field(..., description="Number of twin models available")
    simulations_pending: int = Field(..., description="Number of pending simulations")
    last_simulation_at: Optional[datetime] = Field(None, description="Timestamp of last simulation")


@router.get("/health", response_model=DigitalTwinHealthResponse)
async def digital_twin_health() -> DigitalTwinHealthResponse:
    """
    Health check for Digital Twin service.

    Returns:
        Service health status including model availability and simulation stats.
    """
    # Return sample health data for now
    # In production, this would query actual model and simulation status
    return DigitalTwinHealthResponse(
        status="healthy",
        service="digital-twin",
        models_available=3,
        simulations_pending=0,
        last_simulation_at=datetime.now(timezone.utc),
    )


# =============================================================================
# SIMULATION ENDPOINTS
# =============================================================================


@router.post("/simulate", response_model=SimulationResponse)
async def run_simulation(
    request: SimulateRequest,
    user: Dict[str, Any] = Depends(require_operator),
) -> SimulationResponse:
    """
    Run a digital twin simulation for an intervention.

    Simulates the intervention on a population of digital twins
    and returns predicted Average Treatment Effect (ATE) with recommendation.

    Args:
        request: Simulation parameters including intervention config

    Returns:
        Simulation results with recommendation (deploy/skip/refine)
    """
    from src.digital_twin.models.simulation_models import (
        InterventionConfig,
        PopulationFilter,
    )
    from src.digital_twin.models.twin_models import Brand, TwinType
    from src.digital_twin.simulation_engine import SimulationEngine
    from src.digital_twin.twin_generator import TwinGenerator
    from src.digital_twin.twin_repository import TwinRepository

    logger.info(f"Simulation requested for {request.intervention.intervention_type}")

    try:
        # Build intervention config
        intervention = InterventionConfig(
            intervention_type=request.intervention.intervention_type,
            channel=request.intervention.channel,
            frequency=request.intervention.frequency,
            duration_weeks=request.intervention.duration_weeks,
            content_type=request.intervention.content_type,
            personalization_level=request.intervention.personalization_level,
            target_segment=request.intervention.target_segment,
            target_deciles=request.intervention.target_deciles,
            target_specialties=request.intervention.target_specialties,
            target_regions=request.intervention.target_regions,
            intensity_multiplier=request.intervention.intensity_multiplier,
            extra_params=request.intervention.extra_params,
        )

        # Build population filter
        pop_filter = None
        if request.population_filters:
            pop_filter = PopulationFilter(
                specialties=request.population_filters.specialties,
                deciles=request.population_filters.deciles,
                regions=request.population_filters.regions,
                adoption_stages=request.population_filters.adoption_stages,
                min_baseline_outcome=request.population_filters.min_baseline_outcome,
                max_baseline_outcome=request.population_filters.max_baseline_outcome,
            )

        # Get or create twin population
        twin_type = TwinType(request.twin_type.value)
        brand = Brand(request.brand.value)

        # Initialize generator and generate twins
        generator = TwinGenerator(twin_type=twin_type, brand=brand)
        population = generator.generate(n=request.twin_count)

        # Get model ID (from generator or request)
        model_id = UUID(request.model_id) if request.model_id else generator.model_id or UUID(int=0)

        # Run simulation
        engine = SimulationEngine(
            population=population,
            model_id=model_id,
        )
        result = engine.simulate(
            intervention_config=intervention,
            population_filter=pop_filter,
            calculate_heterogeneity=request.calculate_heterogeneity,
        )

        # Save simulation result
        repo = TwinRepository()
        await repo.save_simulation(result, request.brand.value)

        return SimulationResponse(
            simulation_id=str(result.simulation_id),
            model_id=str(result.model_id),
            intervention_type=intervention.intervention_type,
            brand=request.brand.value,
            twin_type=request.twin_type.value,
            twin_count=result.twin_count,
            simulated_ate=round(result.simulated_ate, 4),
            simulated_ci_lower=round(result.simulated_ci_lower, 4),
            simulated_ci_upper=round(result.simulated_ci_upper, 4),
            simulated_std_error=round(result.simulated_std_error, 4),
            effect_size_cohens_d=result.effect_size_cohens_d,
            statistical_power=result.statistical_power,
            recommendation=RecommendationEnum(result.recommendation.value),
            recommendation_rationale=result.recommendation_rationale,
            recommended_sample_size=result.recommended_sample_size,
            recommended_duration_weeks=result.recommended_duration_weeks,
            simulation_confidence=round(result.simulation_confidence, 3),
            fidelity_warning=result.fidelity_warning,
            fidelity_warning_reason=result.fidelity_warning_reason,
            model_fidelity_score=result.model_fidelity_score,
            status=SimulationStatusEnum(result.status.value),
            error_message=result.error_message,
            execution_time_ms=result.execution_time_ms,
            is_significant=result.is_significant(),
            effect_direction=result.effect_direction(),
            created_at=result.created_at,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/simulations", response_model=SimulationListResponse)
async def list_simulations(
    brand: Optional[BrandEnum] = Query(None, description="Filter by brand"),
    model_id: Optional[str] = Query(None, description="Filter by model ID"),
    status: Optional[SimulationStatusEnum] = Query(None, description="Filter by status"),
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=20, ge=1, le=100, description="Page size"),
) -> SimulationListResponse:
    """
    List simulation results with filtering and pagination.

    Args:
        brand: Optional brand filter
        model_id: Optional model ID filter
        status: Optional status filter
        page: Page number (1-indexed)
        page_size: Results per page

    Returns:
        Paginated list of simulations
    """
    from src.digital_twin.models.simulation_models import SimulationStatus
    from src.digital_twin.twin_repository import TwinRepository

    try:
        repo = TwinRepository()

        # Convert status to SimulationStatus enum if provided
        status_enum = SimulationStatus(status.value) if status else None

        simulations = await repo.list_simulations(
            model_id=UUID(model_id) if model_id else None,
            brand=brand.value if brand else None,
            status=status_enum,
            limit=page_size * page,  # Get enough for pagination
        )

        # Apply pagination manually (repository returns all up to limit)
        offset = (page - 1) * page_size
        paginated = simulations[offset : offset + page_size]

        items = [
            SimulationListItem(
                simulation_id=str(sim.get("simulation_id", "")),
                intervention_type=sim.get("intervention_type", "unknown"),
                brand=sim.get("brand", "unknown"),
                twin_type=sim.get("twin_type", "unknown"),
                twin_count=sim.get("twin_count", 0),
                simulated_ate=round(sim.get("simulated_ate", 0.0), 4),
                recommendation=RecommendationEnum(sim.get("recommendation", "refine")),
                status=SimulationStatusEnum(sim.get("simulation_status", "completed")),
                created_at=sim.get("created_at", datetime.now(timezone.utc)),
            )
            for sim in paginated
        ]

        return SimulationListResponse(
            total_count=len(simulations),
            simulations=items,
            page=page,
            page_size=page_size,
        )

    except Exception as e:
        logger.error(f"Failed to list simulations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/simulations/{simulation_id}", response_model=SimulationDetailResponse)
async def get_simulation(
    simulation_id: str,
) -> SimulationDetailResponse:
    """
    Get detailed information about a simulation.

    Args:
        simulation_id: Simulation UUID

    Returns:
        Detailed simulation result including heterogeneous effects
    """
    from src.digital_twin.twin_repository import TwinRepository

    try:
        repo = TwinRepository()
        result = await repo.get_simulation(UUID(simulation_id))

        if not result:
            raise HTTPException(status_code=404, detail=f"Simulation {simulation_id} not found")

        heterogeneity = EffectHeterogeneityResponse(
            by_specialty=result.effect_heterogeneity.by_specialty,
            by_decile=result.effect_heterogeneity.by_decile,
            by_region=result.effect_heterogeneity.by_region,
            by_adoption_stage=result.effect_heterogeneity.by_adoption_stage,
            top_segments=result.effect_heterogeneity.get_top_segments(5),
        )

        return SimulationDetailResponse(
            simulation_id=str(result.simulation_id),
            model_id=str(result.model_id),
            intervention_type=result.intervention_config.intervention_type,
            brand=result.intervention_config.extra_params.get("brand", "unknown"),
            twin_type=result.intervention_config.extra_params.get("twin_type", "unknown"),
            twin_count=result.twin_count,
            simulated_ate=round(result.simulated_ate, 4),
            simulated_ci_lower=round(result.simulated_ci_lower, 4),
            simulated_ci_upper=round(result.simulated_ci_upper, 4),
            simulated_std_error=round(result.simulated_std_error, 4),
            effect_size_cohens_d=result.effect_size_cohens_d,
            statistical_power=result.statistical_power,
            recommendation=RecommendationEnum(result.recommendation.value),
            recommendation_rationale=result.recommendation_rationale,
            recommended_sample_size=result.recommended_sample_size,
            recommended_duration_weeks=result.recommended_duration_weeks,
            simulation_confidence=round(result.simulation_confidence, 3),
            fidelity_warning=result.fidelity_warning,
            fidelity_warning_reason=result.fidelity_warning_reason,
            model_fidelity_score=result.model_fidelity_score,
            status=SimulationStatusEnum(result.status.value),
            error_message=result.error_message,
            execution_time_ms=result.execution_time_ms,
            is_significant=result.is_significant(),
            effect_direction=result.effect_direction(),
            created_at=result.created_at,
            completed_at=result.completed_at,
            population_filters=result.population_filters.to_dict()
            if result.population_filters
            else {},
            effect_heterogeneity=heterogeneity,
            intervention_config=result.intervention_config.model_dump(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get simulation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# FIDELITY VALIDATION ENDPOINTS
# =============================================================================


@router.post("/validate", response_model=FidelityRecordResponse)
async def validate_simulation(
    request: ValidateFidelityRequest,
    user: Dict[str, Any] = Depends(require_operator),
) -> FidelityRecordResponse:
    """
    Validate a simulation against actual experiment results.

    Updates the fidelity record with actual outcomes and calculates
    prediction error and fidelity grade.

    Args:
        request: Validation data including actual ATE

    Returns:
        Updated fidelity record with grade
    """
    from src.digital_twin.fidelity_tracker import FidelityTracker
    from src.digital_twin.models.simulation_models import SimulationResult
    from src.digital_twin.twin_repository import TwinRepository

    logger.info(f"Validating simulation {request.simulation_id}")

    try:
        repo = TwinRepository()
        tracker = FidelityTracker(repo)

        simulation_uuid = UUID(request.simulation_id)

        # Get the simulation result
        simulation_data = await repo.get_simulation(simulation_uuid)
        if not simulation_data:
            raise HTTPException(
                status_code=404, detail=f"Simulation {request.simulation_id} not found"
            )

        # Check if fidelity record already exists for this simulation
        existing_record = tracker.get_simulation_record(simulation_uuid)

        if not existing_record:
            # Create a minimal SimulationResult to record prediction
            from src.digital_twin.models.simulation_models import (
                InterventionConfig,
                SimulationRecommendation,
            )

            # Build SimulationResult from stored data
            sim_result = SimulationResult(
                simulation_id=simulation_uuid,
                model_id=UUID(simulation_data.get("model_id", str(UUID(int=0)))),
                intervention_config=InterventionConfig(
                    intervention_type=simulation_data.get("intervention_type", "unknown")
                ),
                twin_count=simulation_data.get("twin_count", 0),
                simulated_ate=simulation_data.get("simulated_ate", 0.0),
                simulated_ci_lower=simulation_data.get("simulated_ci_lower", 0.0),
                simulated_ci_upper=simulation_data.get("simulated_ci_upper", 0.0),
                simulated_std_error=simulation_data.get("simulated_std_error", 0.0),
                recommendation=SimulationRecommendation(
                    simulation_data.get("recommendation", "refine")
                ),
                recommendation_rationale=simulation_data.get("recommendation_rationale", ""),
                simulation_confidence=simulation_data.get("simulation_confidence", 0.5),
                execution_time_ms=simulation_data.get("execution_time_ms", 0),
            )

            # Record the prediction
            existing_record = tracker.record_prediction(sim_result)

        # Build CI tuple if both bounds provided
        actual_ci = None
        if request.actual_ci_lower is not None and request.actual_ci_upper is not None:
            actual_ci = (request.actual_ci_lower, request.actual_ci_upper)

        # Validate with actual results
        record = tracker.validate(
            simulation_id=simulation_uuid,
            actual_ate=request.actual_ate,
            actual_ci=actual_ci,
            actual_sample_size=request.actual_sample_size,
            actual_experiment_id=UUID(request.experiment_id) if request.experiment_id else None,
            notes=request.validation_notes,
            confounding_factors=request.confounding_factors,
            validated_by=request.validated_by,
        )

        return FidelityRecordResponse(
            tracking_id=str(record.tracking_id),
            simulation_id=str(record.simulation_id),
            experiment_id=str(record.actual_experiment_id) if record.actual_experiment_id else None,
            simulated_ate=record.simulated_ate,
            simulated_ci_lower=record.simulated_ci_lower,
            simulated_ci_upper=record.simulated_ci_upper,
            actual_ate=record.actual_ate,
            actual_ci_lower=record.actual_ci_lower,
            actual_ci_upper=record.actual_ci_upper,
            actual_sample_size=record.actual_sample_size,
            prediction_error=record.prediction_error,
            absolute_error=record.absolute_error,
            ci_coverage=record.ci_coverage,
            fidelity_grade=FidelityGradeEnum(record.fidelity_grade.value),
            validation_notes=record.validation_notes,
            confounding_factors=record.confounding_factors,
            created_at=record.created_at,
            validated_at=record.validated_at,
            validated_by=record.validated_by,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# MODEL ENDPOINTS
# =============================================================================


@router.get("/models", response_model=ModelListResponse)
async def list_models(
    brand: Optional[BrandEnum] = Query(None, description="Filter by brand"),
    twin_type: Optional[TwinTypeEnum] = Query(None, description="Filter by twin type"),
) -> ModelListResponse:
    """
    List trained twin generator models.

    Args:
        brand: Optional brand filter
        twin_type: Optional twin type filter

    Returns:
        List of active models
    """
    from src.digital_twin.models.twin_models import TwinType
    from src.digital_twin.twin_repository import TwinRepository

    try:
        repo = TwinRepository()

        # Convert twin_type to TwinType enum if provided
        twin_type_enum = TwinType(twin_type.value) if twin_type else None

        models = await repo.list_active_models(
            twin_type=twin_type_enum,
            brand=brand.value if brand else None,
        )

        items = [
            TwinModelSummary(
                model_id=str(m.get("model_id")),
                model_name=m.get("model_name", ""),
                twin_type=m.get("twin_type", ""),
                brand=m.get("brand", ""),
                algorithm=m.get("algorithm", ""),
                r2_score=m.get("r2_score"),
                rmse=m.get("rmse"),
                training_samples=m.get("training_samples", 0),
                is_active=m.get("is_active", True),
                created_at=m.get("created_at", datetime.now(timezone.utc)),
            )
            for m in models
        ]

        return ModelListResponse(
            total_count=len(items),
            models=items,
        )

    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_id}", response_model=TwinModelDetailResponse)
async def get_model(
    model_id: str,
) -> TwinModelDetailResponse:
    """
    Get detailed information about a twin model.

    Args:
        model_id: Model UUID

    Returns:
        Model details including performance metrics
    """
    from src.digital_twin.twin_repository import TwinRepository

    try:
        repo = TwinRepository()
        model = await repo.get_model(UUID(model_id))

        if not model:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

        return TwinModelDetailResponse(
            model_id=str(model.get("model_id")),
            model_name=model.get("model_name", ""),
            model_description=model.get("model_description"),
            twin_type=model.get("twin_type", ""),
            brand=model.get("brand", ""),
            algorithm=model.get("algorithm", ""),
            feature_columns=model.get("feature_columns", []),
            target_column=model.get("target_column", ""),
            r2_score=model.get("r2_score"),
            rmse=model.get("rmse"),
            cv_mean=model.get("cv_mean"),
            cv_std=model.get("cv_std"),
            feature_importances=model.get("feature_importances", {}),
            top_features=model.get("top_features", []),
            training_samples=model.get("training_samples", 0),
            training_duration_seconds=model.get("training_duration_seconds", 0.0),
            is_active=model.get("is_active", True),
            created_at=model.get("created_at", datetime.now(timezone.utc)),
            config=model.get("config", {}),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_id}/fidelity", response_model=FidelityHistoryResponse)
async def get_model_fidelity(
    model_id: str,
    limit: int = Query(default=20, ge=1, le=100, description="Max records"),
    validated_only: bool = Query(default=False, description="Only show validated records"),
) -> FidelityHistoryResponse:
    """
    Get fidelity validation history for a model.

    Args:
        model_id: Model UUID
        limit: Maximum records to return
        validated_only: If True, only return records with actual results

    Returns:
        Fidelity history with grade distribution
    """
    from src.digital_twin.twin_repository import TwinRepository

    try:
        repo = TwinRepository()

        # Get fidelity records for model from repository
        records = await repo.get_model_fidelity_records(
            model_id=UUID(model_id),
            validated_only=validated_only,
            limit=limit,
        )

        # Calculate grade distribution
        grade_dist: Dict[str, int] = {
            "excellent": 0,
            "good": 0,
            "fair": 0,
            "poor": 0,
            "unvalidated": 0,
        }
        total_score = 0.0
        validated_count = 0

        for r in records:
            grade_dist[r.fidelity_grade.value] = grade_dist.get(r.fidelity_grade.value, 0) + 1
            if r.prediction_error is not None:
                # Convert prediction error to fidelity score (1 - |error|)
                fidelity_score = 1.0 - min(abs(r.prediction_error), 1.0)
                total_score += fidelity_score
                validated_count += 1

        avg_score = total_score / validated_count if validated_count > 0 else None

        record_responses = [
            FidelityRecordResponse(
                tracking_id=str(r.tracking_id),
                simulation_id=str(r.simulation_id),
                experiment_id=str(r.actual_experiment_id) if r.actual_experiment_id else None,
                simulated_ate=r.simulated_ate,
                simulated_ci_lower=r.simulated_ci_lower,
                simulated_ci_upper=r.simulated_ci_upper,
                actual_ate=r.actual_ate,
                actual_ci_lower=r.actual_ci_lower,
                actual_ci_upper=r.actual_ci_upper,
                actual_sample_size=r.actual_sample_size,
                prediction_error=r.prediction_error,
                absolute_error=r.absolute_error,
                ci_coverage=r.ci_coverage,
                fidelity_grade=FidelityGradeEnum(r.fidelity_grade.value),
                validation_notes=r.validation_notes,
                confounding_factors=r.confounding_factors,
                created_at=r.created_at,
                validated_at=r.validated_at,
                validated_by=r.validated_by,
            )
            for r in records
        ]

        return FidelityHistoryResponse(
            model_id=model_id,
            total_validations=len(records),
            average_fidelity_score=round(avg_score, 3) if avg_score else None,
            grade_distribution=grade_dist,
            records=record_responses,
        )

    except Exception as e:
        logger.error(f"Failed to get fidelity history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_id}/fidelity/report", response_model=FidelityReportResponse)
async def get_fidelity_report(
    model_id: str,
    lookback_days: int = Query(default=90, ge=7, le=365, description="Days to analyze"),
) -> FidelityReportResponse:
    """
    Get aggregated fidelity report for a model.

    Analyzes fidelity trends and provides degradation warnings.

    Args:
        model_id: Model UUID
        lookback_days: Number of days to look back for analysis

    Returns:
        Fidelity report with trend analysis
    """
    from src.digital_twin.fidelity_tracker import FidelityTracker
    from src.digital_twin.twin_repository import TwinRepository

    try:
        repo = TwinRepository()
        tracker = FidelityTracker(repo)

        # get_model_fidelity_report returns a dict, not an object
        report = tracker.get_model_fidelity_report(UUID(model_id), lookback_days=lookback_days)

        # Extract metrics from the report dict
        metrics = report.get("metrics", {})
        validation_count = report.get("validation_count", 0)
        fidelity_score = report.get("fidelity_score", 0.0)
        ci_coverage_rate = metrics.get("ci_coverage_rate", 0.0)
        is_degrading = report.get("degradation_alert", False)

        # Determine trend based on degradation
        if validation_count == 0:
            trend = "insufficient_data"
            recommendation = "Need more validated predictions"
        elif is_degrading:
            trend = "degrading"
            recommendation = "Consider retraining the twin model"
        elif fidelity_score >= 0.8:
            trend = "excellent"
            recommendation = "Model performing well, continue monitoring"
        elif fidelity_score >= 0.6:
            trend = "stable"
            recommendation = "Model acceptable, monitor for changes"
        else:
            trend = "poor"
            recommendation = "Model performance below threshold, consider retraining"

        return FidelityReportResponse(
            model_id=model_id,
            total_validations=validation_count,
            average_fidelity_score=round(fidelity_score, 3),
            coverage_rate=round(ci_coverage_rate or 0.0, 3),
            grade_distribution=report.get("grade_distribution", {}),
            trend=trend,
            is_degrading=is_degrading,
            degradation_rate=None,  # Could compute from historical data
            recommendation=recommendation,
            generated_at=report.get("computed_at", datetime.now(timezone.utc)),
        )

    except Exception as e:
        logger.error(f"Failed to generate fidelity report: {e}")
        raise HTTPException(status_code=500, detail=str(e))
