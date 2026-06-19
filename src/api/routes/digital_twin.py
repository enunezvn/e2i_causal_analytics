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

import asyncio
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from src.api.dependencies.auth import (
    is_cross_brand_admin,
    require_operator,
    require_viewer,
    resolve_brand_for_read,
)
from src.api.schemas.errors import ErrorResponse, ValidationErrorResponse

if TYPE_CHECKING:
    from src.digital_twin.twin_repository import TwinRepository

logger = logging.getLogger(__name__)

# P2 offload: max seconds the synchronous /simulate endpoint waits for the
# worker_heavy task before returning 408. The frontend axios timeout is 30s
# (frontend/src/lib/api-client.ts); keep this just under it so the client gets a
# clean 408 rather than its own client-side abort. Only consulted when
# HEAVY_OFFLOAD_ENABLED is set; the inline path is unaffected.
_OFFLOAD_TIMEOUT_SECONDS = 28.0

router = APIRouter(
    prefix="/digital-twin",
    tags=["Digital Twin"],
    responses={
        401: {"model": ErrorResponse, "description": "Authentication required"},
        422: {"model": ValidationErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)


async def _get_twin_repo() -> "TwinRepository":
    """Build a TwinRepository backed by a real async Supabase client (fail-closed).

    Before #705 every handler built ``TwinRepository()`` with no client, so all
    sub-repos short-circuited on ``if not self.client`` and twin reads/writes
    were silent no-ops (``twin_simulations`` / ``twin_fidelity_tracking`` stayed
    0-rows on prod). This mirrors the proven monitoring.py pattern
    (``get_async_supabase_client()`` -> repo). ``get_async_supabase_client``
    raises ``ServiceConnectionError`` when the Supabase env is missing — we let
    it surface (fail-closed) rather than silently degrading to a None client.
    """
    from src.digital_twin.twin_repository import TwinRepository
    from src.memory.services.factories import get_async_supabase_client

    client = await get_async_supabase_client()
    return TwinRepository(supabase_client=client)


async def _resolve_active_model_row(
    repo: Any,
    *,
    twin_type: Any,
    brand: Any,
    model_id: Optional[str],
) -> Dict[str, Any]:
    """Resolve the trained model row to simulate with, or fail closed with 503.

    A fresh untrained ``TwinGenerator`` raises ``RuntimeError`` in ``generate()``
    (surfacing as an opaque 500), and a ``UUID(int=0)`` sentinel would violate
    the ``twin_simulations.model_id`` FK. Instead we require a REAL persisted
    model: an explicit ``model_id`` when given, else the highest-fidelity active
    model for the brand/twin_type. ``None`` → honest 503 + ``Retry-After`` (#705 H4).
    """
    if model_id:
        row = await repo.get_model(UUID(model_id))
    else:
        actives = await repo.list_active_models(twin_type=twin_type, brand=brand.value)
        row = actives[0] if actives else None

    if not row:
        raise HTTPException(
            status_code=503,
            detail=(
                f"No trained digital-twin model is available for {brand.value}/"
                f"{twin_type.value}. Train a model before running a simulation."
            ),
            headers={"Retry-After": "30"},
        )
    return cast(Dict[str, Any], row)


async def _load_trained_generator(
    *,
    twin_type: Any,
    brand: Any,
    model_row: Dict[str, Any],
) -> Any:
    """Hydrate a ``TwinGenerator`` from a persisted model row, or fail closed (503).

    The MLflow round-trip (``hydrate_generator``) is synchronous I/O, so it runs
    off the event loop. A load failure is a fail-closed 503 — never a fabricated
    or unscaled-prediction result.
    """
    from src.digital_twin import twin_persistence
    from src.digital_twin.twin_generator import TwinGenerator

    generator = TwinGenerator(twin_type=twin_type, brand=brand)
    loaded = await asyncio.to_thread(
        twin_persistence.hydrate_generator,
        generator,
        model_row.get("mlflow_model_uri"),
        model_row.get("mlflow_run_id"),
    )
    if not loaded:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Trained model {model_row.get('model_id')} for {brand.value}/"
                f"{twin_type.value} could not be loaded from the model registry. "
                "Retry shortly."
            ),
            headers={"Retry-After": "30"},
        )
    return generator


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
        default=1000, ge=100, le=100000, description="Number of twins to simulate"
    )
    confidence_level: float = Field(
        default=0.95,
        ge=0.8,
        le=0.99,
        description=(
            "Confidence level for CI. "
            "(v1: the simulation CI is the estimator's training-evidence 95% interval; "
            "this value is currently not applied)"
        ),
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
                "twin_count": 1000,
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
    data_provenance: Optional[str] = Field(
        default=None,
        description=(
            "Origin of the ATE estimate: 'synthetic_uplift_v1' (synthetic-DGP-trained "
            "uplift, ~constant per brand/intervention in v1) or 'rwd_uplift' (real-world). "
            "None for legacy/error results."
        ),
    )


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
    data_provenance: Optional[str] = None


class SimulationListResponse(BaseModel):
    """Response for listing simulations."""

    total_count: int
    simulations: List[SimulationListItem]
    page: int
    page_size: int


class SimulationHistoryItem(BaseModel):
    """Summary row for the simulation-history view.

    Matches the frontend ``SimulationHistoryResponse.simulations[]`` contract
    (``frontend/src/types/digital-twin.ts``): note ``ate_estimate`` and
    ``recommendation_type`` field names, distinct from ``SimulationListItem``.
    """

    simulation_id: str
    created_at: datetime
    intervention_type: str
    brand: str
    ate_estimate: float
    recommendation_type: str
    data_provenance: Optional[str] = None


class SimulationHistoryResponse(BaseModel):
    """Response for the simulation-history endpoint (frontend contract)."""

    simulations: List[SimulationHistoryItem]
    total: int
    offset: int
    limit: int


class ScenarioSimulateRequest(BaseModel):
    """A single scenario in a comparison request.

    Mirrors the deprecated-but-still-wired frontend ``SimulationRequest`` shape
    so ``compareScenarios`` (frontend/src/api/digital-twin.ts) resolves.
    """

    intervention_type: str
    brand: str
    sample_size: int = Field(default=1000, ge=1)
    duration_days: int = Field(default=90, ge=1)
    twin_type: TwinTypeEnum = Field(default=TwinTypeEnum.HCP)
    twin_count: int = Field(default=1000, ge=100, le=100000)
    target_regions: List[str] = Field(default=[])
    target_segments: List[str] = Field(default=[])
    budget: Optional[float] = Field(default=None)
    parameters: Dict[str, Any] = Field(default={})


class ScenarioComparisonRequest(BaseModel):
    """Request to compare a base scenario against alternatives."""

    base_scenario: ScenarioSimulateRequest
    alternative_scenarios: List[ScenarioSimulateRequest] = Field(default=[])
    comparison_metrics: List[str] = Field(default=[])


class ScenarioComparison(BaseModel):
    """Comparison summary across scenarios."""

    best_scenario_index: int
    metric_comparison: Dict[str, List[float]]
    summary: str


class ScenarioComparisonResult(BaseModel):
    """Response from a scenario comparison run."""

    base_result: SimulationResponse
    alternative_results: List[SimulationResponse]
    comparison: ScenarioComparison


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


@router.get(
    "/health",
    response_model=DigitalTwinHealthResponse,
    summary="Digital Twin service health",
    operation_id="get_digital_twin_health",
)
async def digital_twin_health() -> DigitalTwinHealthResponse:
    """
    Health check for Digital Twin service.

    Reports REAL operational stats sourced from the repository (active model
    count, in-flight simulation count, last simulation timestamp). If the
    repository is unreachable, the service reports ``degraded`` with zeroed
    counts rather than fabricating hardcoded stats.

    Returns:
        Service health status including model availability and simulation stats.
    """

    repo = await _get_twin_repo()

    try:
        models = await repo.list_active_models()
        models_available = len(models)
    except Exception as e:  # repository / DB unreachable
        logger.warning("Digital Twin health: failed to list active models: %s", e)
        return DigitalTwinHealthResponse(
            status="degraded",
            service="digital-twin",
            models_available=0,
            simulations_pending=0,
            last_simulation_at=None,
        )

    pending = 0
    last_simulation_at: Optional[datetime] = None
    try:
        recent = await repo.simulations.list_simulations(limit=100)
        in_flight = {
            SimulationStatusEnum.PENDING.value,
            SimulationStatusEnum.RUNNING.value,
        }
        pending = sum(1 for s in recent if s.get("simulation_status") in in_flight)
        for s in recent:
            created = s.get("created_at")
            if isinstance(created, datetime):
                if last_simulation_at is None or created > last_simulation_at:
                    last_simulation_at = created
    except Exception as e:  # simulations table unreachable — degrade gracefully
        logger.warning("Digital Twin health: failed to list simulations: %s", e)
        return DigitalTwinHealthResponse(
            status="degraded",
            service="digital-twin",
            models_available=models_available,
            simulations_pending=0,
            last_simulation_at=None,
        )

    return DigitalTwinHealthResponse(
        status="healthy",
        service="digital-twin",
        models_available=models_available,
        simulations_pending=pending,
        last_simulation_at=last_simulation_at,
    )


# =============================================================================
# INTERVENTION TAXONOMY ENDPOINT (single source of truth for the dropdown)
# =============================================================================


class InterventionTypeItem(BaseModel):
    """A canonical, selectable intervention type for the simulation dropdown."""

    value: str = Field(..., description="Canonical intervention_type value")
    label: str = Field(..., description="Human-readable label")
    effect_basis: str = Field(
        ...,
        description=(
            "'cohort_causal' (effect is IDENTIFIED in the connected cohort and estimated "
            "by direct DML causal estimation) or 'unavailable' (not identified in the "
            "data — no fabricated effect is produced)"
        ),
    )
    available: bool = Field(
        ...,
        description=(
            "True if a trained twin model exists for the requested brand/twin_type "
            "(else /simulate would 503)."
        ),
    )
    available_for_effect: bool = Field(
        ...,
        description=(
            "True only if the intervention's effect is IDENTIFIED in the connected cohort "
            "(a causal estimate is possible). The frontend should expose only "
            "effect-available interventions; the rest are an honest 'no effect data' "
            "state rather than a fabricated uplift (and /simulate returns 422 for them)."
        ),
    )


class InterventionTypesResponse(BaseModel):
    """Brand-aware list of canonical intervention types for the dropdown."""

    interventions: List[InterventionTypeItem] = Field(default_factory=list)
    brand: Optional[str] = Field(None, description="Brand the availability was resolved for")
    twin_type: str = Field(..., description="Twin type the availability was resolved for")
    timestamp: datetime = Field(..., description="Response timestamp")


@router.get(
    "/intervention-types",
    response_model=InterventionTypesResponse,
    summary="List canonical intervention types (brand-aware availability)",
    operation_id="list_intervention_types",
)
async def list_intervention_types(
    brand: Optional[BrandEnum] = Query(None, description="Resolve availability for this brand"),
    twin_type: TwinTypeEnum = Query(TwinTypeEnum.HCP, description="Twin type"),
    user: Dict[str, Any] = Depends(require_viewer),
) -> InterventionTypesResponse:
    """
    Return the canonical intervention taxonomy — the single source of truth the
    frontend dropdown reads, so FE and backend can never drift.

    Availability is **brand-aware**. ``available`` is True only when a trained twin
    model exists for the brand/twin_type (otherwise ``/simulate`` would 503).
    ``available_for_effect`` is True only when the intervention's effect is IDENTIFIED
    in the connected cohort — i.e. a real causal estimate is possible; the frontend
    should expose only effect-available interventions. ``effect_basis`` is
    ``"cohort_causal"`` for identified interventions (direct DML estimate on the cohort)
    and ``"unavailable"`` otherwise (no fabricated effect; ``/simulate`` returns 422).
    """
    from src.digital_twin.effect.cohort_loader import brand_has_cohort
    from src.digital_twin.effect.provider import (
        COHORT_ESTIMABLE_INTERVENTIONS,
        INTERVENTION_CATALOG,
    )
    from src.digital_twin.models.twin_models import TwinType

    available = False
    has_cohort = False
    if brand is not None:
        try:
            repo = await _get_twin_repo()
            actives = await repo.list_active_models(
                twin_type=TwinType(twin_type.value), brand=brand.value
            )
            available = len(actives) > 0
            # Phase 2: cohort-estimable interventions report effect_basis
            # "cohort_estimated" only when the brand has a usable synthetic-gold
            # cohort to estimate from (else they fall back to the uniform basis).
            has_cohort = await brand_has_cohort(repo.client, brand.value)
        except Exception as e:  # repo/DB unreachable — degrade, never fabricate
            logger.warning("intervention-types: availability/cohort check failed: %s", e)
            available = False
            has_cohort = False

    items = [
        InterventionTypeItem(
            value=value,
            label=label,
            effect_basis=(
                "cohort_causal"
                if (has_cohort and value in COHORT_ESTIMABLE_INTERVENTIONS)
                else "unavailable"
            ),
            available=available,
            available_for_effect=bool(has_cohort and value in COHORT_ESTIMABLE_INTERVENTIONS),
        )
        for value, label in INTERVENTION_CATALOG
    ]
    return InterventionTypesResponse(
        interventions=items,
        brand=brand.value if brand else None,
        twin_type=twin_type.value,
        timestamp=datetime.now(timezone.utc),
    )


# =============================================================================
# SIMULATION ENDPOINTS
# =============================================================================


@router.post(
    "/simulate",
    response_model=SimulationResponse,
    summary="Run twin simulation",
    operation_id="run_digital_twin_simulation",
)
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
    from src.api.dependencies.compute import (
        HeavyComputeSaturated,
        await_celery_result,
        heavy_compute_slot,
        heavy_offload_enabled,
        run_in_bounded_executor,
    )
    from src.digital_twin.models.simulation_models import (
        InterventionConfig,
        PopulationFilter,
        SimulationStatus,
    )
    from src.digital_twin.models.twin_models import Brand, TwinType
    from src.digital_twin.simulation_engine import SimulationEngine

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

        # Resolve a REAL trained model BEFORE generating: an explicit model_id, or
        # the highest-fidelity active model for this brand/twin_type. No model →
        # honest 503 (not a fresh untrained generator → opaque 500, and not a
        # UUID(int=0) sentinel → twin_simulations.model_id FK violation) (#705 H4).
        repo = await _get_twin_repo()
        model_row = await _resolve_active_model_row(
            repo, twin_type=twin_type, brand=brand, model_id=request.model_id
        )
        model_id = UUID(str(model_row["model_id"]))

        # Identification gate (Direction 2): a real causal effect is estimated ONLY for
        # interventions IDENTIFIED in the connected cohort. Build the cohort provider up
        # front; if the intervention is not identified (not a cause in the data, or no
        # usable cohort) the effect is honestly UNAVAILABLE — we never fabricate a
        # synthetic uplift. Gating before the offload/inline split covers both paths.
        from src.digital_twin.effect.cohort_loader import build_cohort_provider_or_none

        cohort_provider = await build_cohort_provider_or_none(
            repo.client, intervention.intervention_type, brand.value
        )
        if cohort_provider is None:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"No effect data available for intervention "
                    f"'{intervention.intervention_type}' and brand '{brand.value}': this "
                    "intervention is not identified in the connected cohort, so a causal "
                    "effect cannot be estimated (no fabricated effect is returned)."
                ),
            )

        if heavy_offload_enabled():
            # P2 offload path (DARK by default): enqueue the heavy compute on
            # worker_heavy and await the result WITHOUT blocking the event loop,
            # preserving the synchronous HTTP contract. The task runs the SAME
            # compute as the inline path (shared src.digital_twin.simulation_runner)
            # and returns a JSON dict we rebuild into the SAME SimulationResult so
            # the response extraction below is byte-identical across both paths.
            # simulation_result_from_dict is a light helper (its heavy imports
            # are function-local), safe to import on the API process.
            from src.digital_twin.simulation_runner import simulation_result_from_dict

            # Enqueue by registered task NAME via the existing send_task idiom
            # (src/workers/celery_app.py) so importing the heavy task package —
            # which pulls sklearn/ML libs into the API process via
            # src/tasks/__init__ — is avoided on the offload path.
            from src.workers.celery_app import celery_app

            payload = {
                "twin_type_value": request.twin_type.value,
                "brand_value": request.brand.value,
                "twin_count": request.twin_count,
                "intervention_dict": intervention.model_dump(mode="json"),
                "population_filter_dict": (
                    pop_filter.model_dump(mode="json") if pop_filter is not None else None
                ),
                "calculate_heterogeneity": request.calculate_heterogeneity,
                "model_id_value": str(model_id),
                # The worker rebuilds a fresh generator, so it must hydrate the
                # SAME persisted model before generating (#705 H4).
                "model_uri": model_row.get("mlflow_model_uri"),
                "model_run_id": model_row.get("mlflow_run_id"),
            }
            async_result = celery_app.send_task(
                "src.tasks.simulate_population", args=[payload], queue="twins"
            )
            try:
                result_dict = await await_celery_result(
                    async_result, timeout=_OFFLOAD_TIMEOUT_SECONDS
                )
            except TimeoutError:
                raise HTTPException(
                    status_code=408,
                    detail="Twin simulation timed out; retry shortly.",
                )
            result = simulation_result_from_dict(result_dict)
        else:
            # P1 inline path (default + fallback). Twin generation + simulation
            # are the heavy, blocking, ~1.3 GiB part of this request. Bound
            # concurrency to one in-flight heavy op per worker (OOM guard) AND run
            # the blocking work off the event loop so it cannot stall the worker.
            # If the per-worker slot budget is exhausted, heavy_compute_slot()
            # raises HeavyComputeSaturated on enter (mapped to a 503 + Retry-After
            # by the app exception handler) — nothing is queued.
            generator = await _load_trained_generator(
                twin_type=twin_type, brand=brand, model_row=model_row
            )

            # Direction 2: estimate the effect DIRECTLY on the brand's cohort via a DML
            # causal estimate (CohortCausalEstimator) over the raw cohort frame supplied
            # by the cohort_provider built above the offload/inline split. No synthetic
            # injected-effect handoff; honest DML inference CI. (Unidentified
            # interventions were already rejected by the identification gate.)
            from src.digital_twin.effect.cohort_causal_estimator import (
                CohortCausalEstimator,
            )

            def _do_sim():
                population = generator.generate(n=request.twin_count)
                engine = SimulationEngine(
                    population=population,
                    effect_provider=cohort_provider,
                    effect_estimator=CohortCausalEstimator(),
                )
                # Pin the resolved DB model id so twin_simulations.model_id FK holds
                # (engine derives self.model_id from population otherwise) (#705 H4).
                engine.model_id = model_id
                return engine.simulate(
                    intervention_config=intervention,
                    population_filter=pop_filter,
                    calculate_heterogeneity=request.calculate_heterogeneity,
                )

            async with heavy_compute_slot():
                result = await run_in_bounded_executor(_do_sim)

        # A FAILED engine result (sub-threshold population / estimation failure)
        # carries ate=0.0 / REFINE and must NOT be surfaced as a 200 success or
        # persisted as a real history row (N1). Fail honestly.
        if result.status == SimulationStatus.FAILED:
            raise HTTPException(
                status_code=422,
                detail=result.error_message or "Simulation could not be completed.",
            )

        # Save simulation result (reuse the repo resolved above).
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
            data_provenance=result.data_provenance,
        )

    except HTTPException:
        # 408 timeout (offload path) must propagate unchanged, not be swallowed
        # into a 500 by the broad handler below.
        raise
    except HeavyComputeSaturated:
        # Reject fast under load — surfaced as 503 + Retry-After by the app
        # exception handler. Must precede the broad handlers so it is not
        # swallowed into a 500.
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise HTTPException(status_code=500, detail="Simulation failed")


@router.get(
    "/simulations",
    response_model=SimulationListResponse,
    summary="List simulations",
    operation_id="list_twin_simulations",
)
async def list_simulations(
    brand: Optional[BrandEnum] = Query(None, description="Filter by brand"),
    model_id: Optional[str] = Query(None, description="Filter by model ID"),
    status: Optional[SimulationStatusEnum] = Query(None, description="Filter by status"),
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=20, ge=1, le=100, description="Page size"),
    user: Dict[str, Any] = Depends(require_viewer),
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

    # Fail-closed brand scoping (H11): a non-admin caller may only read brands in
    # their grant; admin / ['all'] is unaffected. Never leave the read unscoped.
    allowed, effective_brand = resolve_brand_for_read(user, brand.value if brand else None)
    if not allowed:
        raise HTTPException(status_code=403, detail="Brand not permitted for this user.")

    try:
        repo = await _get_twin_repo()

        # Convert status to SimulationStatus enum if provided
        status_enum = SimulationStatus(status.value) if status else None

        simulations = await repo.simulations.list_simulations(
            model_id=UUID(model_id) if model_id else None,
            brand=effective_brand,
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
                data_provenance=sim.get("data_provenance"),
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
        raise HTTPException(status_code=500, detail="Failed to list simulations")


# NOTE: the literal /simulations/history and /simulations/compare routes MUST be
# declared BEFORE the dynamic /simulations/{simulation_id} route below.
# FastAPI matches routes in declaration order; otherwise "history"/"compare"
# would be captured as a simulation_id and UUID(...) would raise → 500/404.
@router.get(
    "/simulations/history",
    response_model=SimulationHistoryResponse,
    summary="Simulation history",
    operation_id="get_simulation_history",
)
async def get_simulation_history(
    limit: int = Query(default=20, ge=1, le=100, description="Max records to return"),
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
    user: Dict[str, Any] = Depends(require_viewer),
) -> SimulationHistoryResponse:
    """
    Return recent simulation history for the dashboard.

    Sourced from the real simulation repository. Maps stored rows to the
    frontend ``SimulationHistoryResponse`` contract (``ate_estimate`` /
    ``recommendation_type``).

    Args:
        limit: Maximum number of records to return.
        offset: Pagination offset.

    Returns:
        Simulation history rows with total count and pagination echo.
    """

    # Fail-closed brand scoping (H11): no brand filter param here, so a non-admin
    # is pinned to their first granted brand; admin / ['all'] sees all.
    allowed, effective_brand = resolve_brand_for_read(user, None)
    if not allowed:
        raise HTTPException(status_code=403, detail="No brand grant for this user.")

    try:
        repo = await _get_twin_repo()
        # Fetch enough rows to cover the requested window (repo returns newest
        # first), then apply the offset/limit slice.
        rows = await repo.simulations.list_simulations(brand=effective_brand, limit=offset + limit)
        window = rows[offset : offset + limit]

        items = [
            SimulationHistoryItem(
                simulation_id=str(sim.get("simulation_id", "")),
                created_at=sim.get("created_at", datetime.now(timezone.utc)),
                intervention_type=sim.get("intervention_type", "unknown"),
                brand=sim.get("brand", "unknown"),
                ate_estimate=round(sim.get("simulated_ate", 0.0), 4),
                recommendation_type=sim.get("recommendation", "refine"),
                data_provenance=sim.get("data_provenance"),
            )
            for sim in window
        ]

        return SimulationHistoryResponse(
            simulations=items,
            total=len(rows),
            offset=offset,
            limit=limit,
        )

    except Exception as e:
        logger.error(f"Failed to get simulation history: {e}")
        raise HTTPException(status_code=500, detail="Failed to get simulation history")


@router.post(
    "/simulations/compare",
    response_model=ScenarioComparisonResult,
    summary="Compare simulation scenarios",
    operation_id="compare_twin_scenarios",
)
async def compare_scenarios(
    request: ScenarioComparisonRequest,
    user: Dict[str, Any] = Depends(require_operator),
) -> ScenarioComparisonResult:
    """
    Run a base scenario plus alternatives and return a comparison.

    Each scenario is executed through the same TwinGenerator + SimulationEngine
    used by ``/simulate``, so results are real (not fabricated). The scenario
    with the largest simulated ATE is reported as ``best_scenario_index`` (0 =
    base scenario).

    Args:
        request: Base scenario and alternative scenarios to compare.

    Returns:
        Base + alternative results with a comparison summary.
    """
    from src.api.dependencies.compute import (
        HeavyComputeSaturated,
        heavy_compute_slot,
        run_in_bounded_executor,
    )
    from src.digital_twin.models.simulation_models import InterventionConfig
    from src.digital_twin.models.twin_models import Brand, TwinType
    from src.digital_twin.simulation_engine import SimulationEngine

    logger.info(
        "Scenario comparison requested: base=%s alternatives=%d",
        request.base_scenario.intervention_type,
        len(request.alternative_scenarios),
    )

    async def _load_for(scenario: ScenarioSimulateRequest) -> Any:
        # Each scenario simulates against its own brand/twin_type trained model; a
        # scenario with no loadable model fails the whole comparison closed (503),
        # rather than generating from an untrained generator (#705 H4).
        twin_type = TwinType(scenario.twin_type.value)
        brand = Brand(scenario.brand)
        model_row = await _resolve_active_model_row(
            repo,
            twin_type=twin_type,
            brand=brand,
            model_id=getattr(scenario, "model_id", None),
        )
        generator = await _load_trained_generator(
            twin_type=twin_type, brand=brand, model_row=model_row
        )
        return generator, model_row

    def _run_scenario(
        scenario: ScenarioSimulateRequest, generator: Any, model_row: Dict[str, Any]
    ) -> SimulationResponse:
        intervention = InterventionConfig(
            intervention_type=scenario.intervention_type,
            target_regions=scenario.target_regions,
            extra_params={
                "brand": scenario.brand,
                "twin_type": scenario.twin_type.value,
                "sample_size": scenario.sample_size,
                "duration_days": scenario.duration_days,
                "budget": scenario.budget,
                **scenario.parameters,
            },
        )
        # Use the resolved DB model_id — never a UUID(int=0) sentinel (#705 H4).
        model_id = UUID(str(model_row["model_id"]))
        population = generator.generate(n=scenario.twin_count)
        engine = SimulationEngine(population=population)
        engine.model_id = model_id
        result = engine.simulate(intervention_config=intervention)
        # A failed scenario carries ate=0.0 / REFINE — fail the comparison closed
        # rather than reporting a fake zero-effect scenario (N1). result.status is
        # the domain SimulationStatus enum; compare on its value (SimulationStatus
        # is not imported in compare_scenarios).
        if result.status.value == "failed":
            raise HTTPException(
                status_code=422,
                detail=result.error_message or "A scenario simulation could not be completed.",
            )

        return SimulationResponse(
            simulation_id=str(result.simulation_id),
            model_id=str(result.model_id),
            intervention_type=intervention.intervention_type,
            brand=scenario.brand,
            twin_type=scenario.twin_type.value,
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
            data_provenance=result.data_provenance,
        )

    try:
        repo = await _get_twin_repo()
        base_gen, base_row = await _load_for(request.base_scenario)
        alt_loaded = [await _load_for(s) for s in request.alternative_scenarios]

        # Twin generation is the heavy, blocking, ~1.3 GiB work. Run every scenario
        # off the event loop under ONE per-worker heavy-compute slot (mirrors the
        # /simulate inline path) so a multi-scenario compare can't stall the worker
        # or bypass the OOM budget the slot enforces.
        async with heavy_compute_slot():
            base_result = await run_in_bounded_executor(
                _run_scenario, request.base_scenario, base_gen, base_row
            )
            alternative_results = [
                await run_in_bounded_executor(_run_scenario, s, gen, row)
                for s, (gen, row) in zip(request.alternative_scenarios, alt_loaded, strict=True)
            ]

        all_results = [base_result, *alternative_results]
        ates = [r.simulated_ate for r in all_results]
        best_index = max(range(len(ates)), key=lambda i: ates[i])

        comparison = ScenarioComparison(
            best_scenario_index=best_index,
            metric_comparison={"simulated_ate": ates},
            summary=(
                f"Scenario {best_index} has the largest simulated ATE ({ates[best_index]:.4f})."
            ),
        )

        return ScenarioComparisonResult(
            base_result=base_result,
            alternative_results=alternative_results,
            comparison=comparison,
        )

    except HTTPException:
        # Honest 503 (no/unloadable model) must propagate, not collapse to 500.
        raise
    except HeavyComputeSaturated:
        # Reject fast under load (mapped to 503 + Retry-After by the app handler).
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Scenario comparison failed: {e}")
        raise HTTPException(status_code=500, detail="Scenario comparison failed")


@router.get(
    "/simulations/{simulation_id}",
    response_model=SimulationDetailResponse,
    summary="Get simulation details",
    operation_id="get_twin_simulation",
)
async def get_simulation(
    simulation_id: str,
    user: Dict[str, Any] = Depends(require_viewer),
) -> SimulationDetailResponse:
    """
    Get detailed information about a simulation.

    Args:
        simulation_id: Simulation UUID

    Returns:
        Detailed simulation result including heterogeneous effects
    """

    try:
        repo = await _get_twin_repo()
        result = await repo.get_simulation(UUID(simulation_id))

        if not result:
            raise HTTPException(status_code=404, detail=f"Simulation {simulation_id} not found")

        # repo.get_simulation returns the RAW twin_simulations row (a dict), not a
        # SimulationResult object. The prior handler accessed it as an object
        # (result.simulation_id / .is_significant()) under # type: ignore[attr-defined],
        # which 500'd on every real row — masked only because twin_simulations was
        # 0 rows. R1 makes rows persist, so this is now a live bug (#705 H5b/H11).
        # Map from the dict; derive the fields the row does not persist.
        ci_lower = float(result.get("simulated_ci_lower", 0.0) or 0.0)
        ci_upper = float(result.get("simulated_ci_upper", 0.0) or 0.0)
        ate = float(result.get("simulated_ate", 0.0) or 0.0)
        is_significant = not (ci_lower <= 0.0 <= ci_upper)
        effect_direction = "positive" if ate > 0 else "negative" if ate < 0 else "neutral"

        # Fail-closed ownership check (H11): a non-admin may only read a simulation
        # whose brand is in their grant. 404 (not 403) so we don't leak existence;
        # deny non-admins when the brand can't be determined (fail-closed).
        sim_brand = result.get("brand")
        if not is_cross_brand_admin(user) and (
            sim_brand is None or not resolve_brand_for_read(user, sim_brand)[0]
        ):
            raise HTTPException(status_code=404, detail=f"Simulation {simulation_id} not found")

        eh = result.get("effect_heterogeneity") or {}
        heterogeneity = EffectHeterogeneityResponse(
            by_specialty=eh.get("by_specialty", {}),
            by_decile=eh.get("by_decile", {}),
            by_region=eh.get("by_region", {}),
            by_adoption_stage=eh.get("by_adoption_stage", {}),
            top_segments=(eh.get("top_segments") or [])[:5],
        )

        return SimulationDetailResponse(
            simulation_id=str(result.get("simulation_id", "")),
            model_id=str(result.get("model_id", "")),
            intervention_type=result.get("intervention_type", "unknown"),
            brand=result.get("brand", "unknown"),
            # twin_type is not persisted on the row — mirror list_simulations' default.
            twin_type=result.get("twin_type", "unknown"),
            twin_count=result.get("twin_count", 0),
            simulated_ate=round(ate, 4),
            simulated_ci_lower=round(ci_lower, 4),
            simulated_ci_upper=round(ci_upper, 4),
            simulated_std_error=round(float(result.get("simulated_std_error", 0.0) or 0.0), 4),
            effect_size_cohens_d=result.get("effect_size_cohens_d"),
            statistical_power=result.get("statistical_power"),
            recommendation=RecommendationEnum(result.get("recommendation", "refine")),
            recommendation_rationale=result.get("recommendation_rationale", ""),
            recommended_sample_size=result.get("recommended_sample_size"),
            recommended_duration_weeks=result.get("recommended_duration_weeks"),
            simulation_confidence=round(float(result.get("simulation_confidence", 0.0) or 0.0), 3),
            fidelity_warning=bool(result.get("fidelity_warning", False)),
            fidelity_warning_reason=result.get("fidelity_warning_reason"),
            model_fidelity_score=result.get("model_fidelity_score"),
            status=SimulationStatusEnum(result.get("simulation_status", "completed")),
            error_message=result.get("error_message"),
            execution_time_ms=result.get("execution_time_ms", 0),
            is_significant=is_significant,
            effect_direction=effect_direction,
            created_at=result.get("created_at", datetime.now(timezone.utc)),
            completed_at=result.get("completed_at"),
            population_filters=result.get("population_filters") or {},
            effect_heterogeneity=heterogeneity,
            intervention_config=result.get("intervention_config") or {},
            data_provenance=result.get("data_provenance"),  # #705 H5b
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get simulation: {e}")
        raise HTTPException(status_code=500, detail="Failed to get simulation")


# =============================================================================
# FIDELITY VALIDATION ENDPOINTS
# =============================================================================


@router.post(
    "/validate",
    response_model=FidelityRecordResponse,
    summary="Validate simulation fidelity",
    operation_id="validate_twin_simulation",
)
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

    logger.info(f"Validating simulation {request.simulation_id}")

    try:
        repo = await _get_twin_repo()
        tracker = FidelityTracker(repo)

        simulation_uuid = UUID(request.simulation_id)

        # Get the simulation result
        simulation_data = await repo.get_simulation(simulation_uuid)
        if not simulation_data:
            raise HTTPException(
                status_code=404, detail=f"Simulation {request.simulation_id} not found"
            )

        # Check if fidelity record already exists for this simulation
        existing_record = await tracker.get_simulation_record(simulation_uuid)

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
            existing_record = await tracker.record_prediction(sim_result)

        # Build CI tuple if both bounds provided
        actual_ci = None
        if request.actual_ci_lower is not None and request.actual_ci_upper is not None:
            actual_ci = (request.actual_ci_lower, request.actual_ci_upper)

        # Validate with actual results
        record = await tracker.validate(
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
        raise HTTPException(status_code=500, detail="Validation failed")


# =============================================================================
# MODEL ENDPOINTS
# =============================================================================


@router.get(
    "/models",
    response_model=ModelListResponse,
    summary="List twin models",
    operation_id="list_twin_models",
)
async def list_models(
    brand: Optional[BrandEnum] = Query(None, description="Filter by brand"),
    twin_type: Optional[TwinTypeEnum] = Query(None, description="Filter by twin type"),
    user: Dict[str, Any] = Depends(require_viewer),
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

    # Fail-closed brand scoping (H11): a non-admin only sees their granted
    # brand's models; admin / ['all'] sees all.
    allowed, effective_brand = resolve_brand_for_read(user, brand.value if brand else None)
    if not allowed:
        raise HTTPException(status_code=403, detail="Brand not permitted for this user.")

    try:
        repo = await _get_twin_repo()

        # Convert twin_type to TwinType enum if provided
        twin_type_enum = TwinType(twin_type.value) if twin_type else None

        models = await repo.list_active_models(
            twin_type=twin_type_enum,
            brand=effective_brand,
        )

        # save_model stores metrics nested under performance_metrics (JSONB) and
        # tuning under training_config (JSONB) — NOT as flat columns. Read from
        # the nested dicts (with a flat fallback for the v_active_twin_models view
        # / legacy rows) so real trained models are not shown metric-less (#705 H4).
        items = []
        for m in models:
            pm = m.get("performance_metrics") or {}
            tc = m.get("training_config") or {}
            items.append(
                TwinModelSummary(
                    model_id=str(m.get("model_id")),
                    model_name=m.get("model_name", ""),
                    twin_type=m.get("twin_type", ""),
                    brand=m.get("brand", ""),
                    algorithm=tc.get("algorithm", m.get("algorithm", "")),
                    r2_score=pm.get("r2_score", m.get("r2_score")),
                    rmse=pm.get("rmse", m.get("rmse")),
                    training_samples=tc.get(
                        "training_samples",
                        pm.get("training_samples", m.get("training_samples", 0)),
                    ),
                    is_active=m.get("is_active", True),
                    created_at=m.get("created_at", datetime.now(timezone.utc)),
                )
            )

        return ModelListResponse(
            total_count=len(items),
            models=items,
        )

    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail="Failed to list models")


@router.get(
    "/models/{model_id}",
    response_model=TwinModelDetailResponse,
    summary="Get twin model details",
    operation_id="get_twin_model",
)
async def get_model(
    model_id: str,
    user: Dict[str, Any] = Depends(require_viewer),
) -> TwinModelDetailResponse:
    """
    Get detailed information about a twin model.

    Args:
        model_id: Model UUID

    Returns:
        Model details including performance metrics
    """

    try:
        repo = await _get_twin_repo()
        model = await repo.get_model(UUID(model_id))

        if not model:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

        # Fail-closed ownership check (H11): a non-admin may only read a model
        # whose brand is in their grant. 404 (not 403) so existence is not leaked;
        # deny when the brand cannot be determined. Mirrors get_simulation/{id}.
        model_brand = model.get("brand")
        if not is_cross_brand_admin(user) and (
            model_brand is None or not resolve_brand_for_read(user, model_brand)[0]
        ):
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

        # Read from the nested JSONB columns save_model actually writes
        # (performance_metrics / training_config / target_columns), with a flat
        # fallback for view/legacy rows (#705 H4).
        pm = model.get("performance_metrics") or {}
        tc = model.get("training_config") or {}
        target_cols = model.get("target_columns") or []
        return TwinModelDetailResponse(
            model_id=str(model.get("model_id")),
            model_name=model.get("model_name", ""),
            model_description=model.get("model_description"),
            twin_type=model.get("twin_type", ""),
            brand=model.get("brand", ""),
            algorithm=tc.get("algorithm", model.get("algorithm", "")),
            feature_columns=model.get("feature_columns", []),
            target_column=(target_cols[0] if target_cols else model.get("target_column", "")),
            r2_score=pm.get("r2_score", model.get("r2_score")),
            rmse=pm.get("rmse", model.get("rmse")),
            cv_mean=pm.get("cv_mean", model.get("cv_mean")),
            cv_std=pm.get("cv_std", model.get("cv_std")),
            feature_importances=pm.get("feature_importances", model.get("feature_importances", {})),
            top_features=pm.get("top_features", model.get("top_features", [])),
            training_samples=tc.get(
                "training_samples",
                pm.get("training_samples", model.get("training_samples", 0)),
            ),
            training_duration_seconds=pm.get("training_duration_seconds", 0.0),
            is_active=model.get("is_active", True),
            created_at=model.get("created_at", datetime.now(timezone.utc)),
            config=tc or model.get("config", {}),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model: {e}")
        raise HTTPException(status_code=500, detail="Failed to get model")


@router.get("/models/{model_id}/fidelity", response_model=FidelityHistoryResponse)
async def get_model_fidelity(
    model_id: str,
    limit: int = Query(default=20, ge=1, le=100, description="Max records"),
    validated_only: bool = Query(default=False, description="Only show validated records"),
    user: Dict[str, Any] = Depends(require_viewer),
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

    try:
        repo = await _get_twin_repo()

        # Resolve the model first so the read is fail-closed brand-scoped (H11)
        # and 404s honestly on an unknown model id.
        model = await repo.get_model(UUID(model_id))
        if not model:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        model_brand = model.get("brand")
        if not is_cross_brand_admin(user) and (
            model_brand is None or not resolve_brand_for_read(user, model_brand)[0]
        ):
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

        # Get fidelity records for model from repository
        records = await repo.get_model_fidelity_records(  # type: ignore[attr-defined]
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

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get fidelity history: {e}")
        raise HTTPException(status_code=500, detail="Failed to get fidelity history")


@router.get("/models/{model_id}/fidelity/report", response_model=FidelityReportResponse)
async def get_fidelity_report(
    model_id: str,
    lookback_days: int = Query(default=90, ge=7, le=365, description="Days to analyze"),
    user: Dict[str, Any] = Depends(require_viewer),
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

    try:
        repo = await _get_twin_repo()

        # Resolve the model first so the read is fail-closed brand-scoped (H11)
        # and 404s honestly on an unknown model id.
        model = await repo.get_model(UUID(model_id))
        if not model:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        model_brand = model.get("brand")
        if not is_cross_brand_admin(user) and (
            model_brand is None or not resolve_brand_for_read(user, model_brand)[0]
        ):
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

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

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate fidelity report: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate fidelity report")
