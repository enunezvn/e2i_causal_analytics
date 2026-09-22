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
import hashlib
import json
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, cast
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from src.api.dependencies.auth import (
    is_cross_brand_admin,
    require_operator,
    require_viewer,
    resolve_brand_for_read,
)
from src.api.routes.digital_twin_capability import (
    effect_data_unmeasured,
    simulable_brands,
    warn_model_without_effect_data,
)
from src.api.routes.digital_twin_rejections import Decile, rejected_request
from src.api.schemas.digital_twin import (
    DigitalTwinHealthResponse,
    EffectHeterogeneityResponse,
    InterventionTypeItem,
    InterventionTypesResponse,
)
from src.api.schemas.digital_twin import heterogeneity_response as _heterogeneity_response
from src.api.schemas.digital_twin import live_subgroups_basis as _live_subgroups_basis
from src.api.schemas.errors import ErrorResponse, ValidationErrorResponse

if TYPE_CHECKING:
    from src.digital_twin.twin_repository import StoredEstimateScope, TwinRepository

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


async def _verify_experiment_link(repo: Any, experiment_id: UUID, brand: str) -> None:
    """The experiment a pre-screen links to must exist and be this brand's (#2206).

    ``twin_simulations.experiment_design_id`` has no FK, so nothing else would
    catch a typo'd or foreign-brand id; the fidelity producer resolves the twin
    simulation through this link and would only ever skip.
    """
    res = await (
        repo.client.table("ml_experiments")
        .select("id,brand")
        .eq("id", str(experiment_id))
        .limit(1)
        .execute()
    )
    rows = res.data or []
    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"experiment_design_id {experiment_id} does not name an existing experiment.",
        )
    exp_brand = rows[0].get("brand")
    if exp_brand and str(exp_brand) != str(brand):
        raise HTTPException(
            status_code=422,
            detail=(
                f"experiment_design_id {experiment_id} belongs to brand {exp_brand}, "
                f"not {brand}; a pre-screen links only to its own brand's experiment."
            ),
        )


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


class EstimateScopeEnum(str, Enum):
    """What a simulation's effect was estimated ON (#2053)."""

    COHORT = "cohort"
    REGIONS = "regions"
    UNKNOWN = "unknown"


def _live_estimate_scope(target_regions: List[str]) -> EstimateScopeEnum:
    """Scope of a simulation this request just ran: it always knows its own."""
    return EstimateScopeEnum.REGIONS if target_regions else EstimateScopeEnum.COHORT


def _stored_estimate_scope(row: Dict[str, Any]) -> "StoredEstimateScope":
    """Scope recorded on a stored twin_simulations row; unknown when none was (#2053)."""
    from src.digital_twin.twin_repository import StoredEstimateScope

    return StoredEstimateScope.from_row(row)


def _round4(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(float(value), 4)


# ---------------------------------------------------------------------------
# #2206 — honest model surfacing. Every statement below is DERIVED from the rows
# (fit fingerprints, feature columns, provenance, fidelity columns), never hardcoded.
# ---------------------------------------------------------------------------

# Wall-clock is not part of the fit: two runs of one deterministic training
# (same frame, seed, config, features) differ only here.
_FIT_FINGERPRINT_EXCLUDED_METRICS = frozenset({"training_duration_seconds"})


def _canonical(value: Any) -> Any:
    """JSONB round-trips ``1`` and ``1.0`` interchangeably; hash them the same.

    Integral floats become ints (lossless); ints are never widened to float, so
    values above 2**53 keep their identity.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        # Integral floats fold to int; other floats stay NUMERIC (json.dumps emits
        # repr), so 0.2 never collides with the string "0.2" (codex r9 #2).
        return int(value) if value.is_integer() else value
    if isinstance(value, int):
        return value
    if isinstance(value, dict):
        return {str(k): _canonical(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    return value


def _fit_fingerprint(row: Dict[str, Any]) -> str:
    """A content hash of the RECORDED fit: training_config (including the
    ``training_frame`` identity — source/seed/rows — when the trainer recorded it),
    feature/target columns, and every reported metric except wall-clock.

    Equal fingerprints = the same recorded fit. It does not hash the artifact
    itself; rows trained before ``training_frame`` was recorded are compared on
    config + columns + full-precision metrics (prod: three brand rows, one seed-0
    synthetic frame, identical R²/CV/importances to 16 digits).
    """
    pm = dict(row.get("performance_metrics") or {})
    for key in _FIT_FINGERPRINT_EXCLUDED_METRICS:
        pm.pop(key, None)
    payload = _canonical(
        {
            "training_config": row.get("training_config") or {},
            "feature_columns": list(row.get("feature_columns") or []),
            "target_columns": list(row.get("target_columns") or []),
            "performance_metrics": pm,
        }
    )
    canonical = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _model_fidelity_state(
    row: Dict[str, Any],
) -> "tuple[FidelityStatusEnum, Optional[float], int]":
    """(status, fidelity_score, fidelity_sample_count) from the model row's columns.

    Defined ahead of the enum classes below; the annotation is a forward reference.
    """
    from src.digital_twin.models.simulation_models import classify_fidelity

    score = row.get("fidelity_score")
    score_f = None if score is None else float(score)
    status, _warn, _reason = classify_fidelity(score_f)
    return FidelityStatusEnum(status.value), score_f, int(row.get("fidelity_sample_count") or 0)


def _stored_fidelity_fields(model_row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Fidelity fields for a STORED simulation, derived from its model row now (#2206).

    twin_simulations persists only the gate's verdict of its time — and that gate
    let a NULL score pass as ``fidelity_warning=False``. Reading the model row
    through the same rule the engine now uses keeps the stored history honest:
    an unvalidated model warns, a validated one does not.
    """
    from src.digital_twin.models.simulation_models import classify_fidelity

    score = (model_row or {}).get("fidelity_score")
    score_f = None if score is None else float(score)
    status, warning, reason = classify_fidelity(score_f)
    return {
        "fidelity_status": FidelityStatusEnum(status.value),
        "model_fidelity_score": score_f,
        "fidelity_warning": warning,
        "fidelity_warning_reason": reason,
    }


def _r2_score_basis(data_provenance: Optional[str]) -> "R2ScoreBasisEnum":
    if not data_provenance:
        return R2ScoreBasisEnum.UNKNOWN
    prov = str(data_provenance).lower()
    if prov.startswith("synthetic"):
        return R2ScoreBasisEnum.SYNTHETIC_TARGET
    if prov.startswith("rwd"):
        return R2ScoreBasisEnum.RWD_TARGET
    return R2ScoreBasisEnum.UNKNOWN


def _model_honesty_fields(
    row: Dict[str, Any],
    census: List[Dict[str, Any]],
    user: Dict[str, Any],
) -> Dict[str, Any]:
    """The #2206 fields for one model row, given the census of ALL active rows of
    its twin_type. ``shared_fit_model_count`` is the data-derived fact; the other
    brands' names are given only when the caller may read those brands (H11)."""
    tc = row.get("training_config") or {}
    provenance = tc.get("data_provenance", row.get("data_provenance"))
    fingerprint = _fit_fingerprint(row)
    # Distinct BRANDS sharing the fingerprint (codex r5 #2): nothing enforces one
    # active row per brand, and two active versions of one brand are one label.
    same_fit_brands = {
        str(r.get("brand"))
        for r in census
        if r.get("brand")
        and _fit_fingerprint(r) == fingerprint
        and str(r.get("twin_type", "")) == str(row.get("twin_type", ""))
    }
    own_brand = str(row.get("brand")) if row.get("brand") else None
    if own_brand:
        same_fit_brands.add(own_brand)
    others = sorted(b for b in same_fit_brands if b != own_brand)
    visible_others = [b for b in others if resolve_brand_for_read(user, b)[0]]
    status, score, n = _model_fidelity_state(row)
    features = [str(c) for c in (row.get("feature_columns") or [])]
    frame_meta = tc.get("training_frame") or {}
    return {
        # A content digest is the frame's identity; source/seed/path alone are not.
        "training_frame_recorded": bool(frame_meta.get("content_sha256")),
        "fidelity_status": status,
        "fidelity_score": score,
        "fidelity_sample_count": n,
        "data_provenance": provenance,
        "r2_score_basis": _r2_score_basis(provenance),
        "brand_is_feature": "brand" in {f.lower() for f in features},
        "training_fingerprint": fingerprint,
        "shared_fit_model_count": max(1, len(same_fit_brands)),
        "shared_fit_with": visible_others,
    }


# list_active_models defaults to 100 rows; a census that stopped there would
# silently drop fits. Ask for far more than any real registry and say if it is hit.
_CENSUS_LIMIT = 10_000


async def _active_model_census(repo: Any, twin_type_enum: Any) -> List[Dict[str, Any]]:
    """Every active model of the twin_type, across brands — the shared-fit census.

    Brand scoping is applied to the LISTING afterwards; the census must see all
    brands or a single-brand caller could never learn that their fit is shared.
    """
    rows = list(
        await repo.list_active_models(twin_type=twin_type_enum, brand=None, limit=_CENSUS_LIMIT)
        or []
    )
    if len(rows) >= _CENSUS_LIMIT:
        logger.warning(
            "Active twin-model census hit its limit (%d rows); shared-fit counts may omit rows",
            _CENSUS_LIMIT,
        )
    return rows


class FidelityStatusEnum(str, Enum):
    """Whether the model's fidelity has ever been measured (#2206).

    'unvalidated': digital_twin_models.fidelity_score is NULL — no experiment outcome
    has been compared against the model; a NULL never reads as "passed".
    'below_threshold': measured and below the engine's 0.70 gate.
    'validated': measured, at or above the gate.
    """

    UNVALIDATED = "unvalidated"
    BELOW_THRESHOLD = "below_threshold"
    VALIDATED = "validated"


class R2ScoreBasisEnum(str, Enum):
    """What the model's r2_score was scored against (#2206).

    'synthetic_target': the fit's target is self-generated by
    synthetic_training_frame (data_provenance 'synthetic') — the R² says how well the
    model reproduces its own synthetic generator, not real-world outcomes.
    'rwd_target': trained on a real-world data file.
    """

    SYNTHETIC_TARGET = "synthetic_target"
    RWD_TARGET = "rwd_target"
    UNKNOWN = "unknown"


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
    target_deciles: List[Decile] = Field(default=[1, 2, 3], description="Target deciles (1-10)")
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
    fidelity_status: FidelityStatusEnum = Field(
        description=(
            "Explicit fidelity state of the model behind this run (#2206): "
            "'unvalidated' when its fidelity_score is NULL (no experiment outcome has "
            "been compared against it — fidelity_warning is True and this is NOT a "
            "pass), 'below_threshold' or 'validated' when measured. A stored "
            "simulation derives it from the model row at read time."
        ),
    )
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
    estimate_scope: EstimateScopeEnum = Field(
        description=(
            "What simulated_ate and its interval were estimated ON (#2053): 'cohort' (the "
            "whole cohort), 'regions' (target_regions), or 'unknown' (a stored simulation "
            "written before the scope was recorded, whose effect may be either). An empty "
            "target_regions means cohort-wide only when this is 'cohort'."
        ),
    )
    target_regions: List[str] = Field(
        default=[],
        description=(
            "Regions the effect above was estimated ON (#2023). Empty unless estimate_scope "
            "is 'regions'. When a region filter is applied, simulated_ate / its interval / the "
            "recommendation / recommended_sample_size all describe these regions — the "
            "same numbers the chat counterfactual_simulator gives for the same question."
        ),
    )
    cohort_effect: Optional[float] = Field(
        default=None,
        description=(
            "The cohort-wide ATE the targeted estimate was narrowed from, reported "
            "alongside it. None unless estimate_scope is 'regions'."
        ),
    )
    cohort_ci_lower: Optional[float] = Field(
        default=None, description="Lower bound of the cohort-wide interval, when narrowed."
    )
    cohort_ci_upper: Optional[float] = Field(
        default=None, description="Upper bound of the cohort-wide interval, when narrowed."
    )
    effect_heterogeneity: EffectHeterogeneityResponse = Field(
        description="Supported subgroup effects and their evidence provenance."
    )
    subgroups_basis: Literal["cohort_rows", "per_twin", "twin_weighted_legacy", "unknown"] = Field(
        description="How effect_heterogeneity was computed; fresh cohort runs use cohort_rows.",
    )


class SimulationDetailResponse(SimulationResponse):
    """Detailed simulation response including heterogeneity."""

    population_filters: Dict[str, Any]
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
    # Scope of simulated_ate (#2053); see SimulationResponse.estimate_scope.
    estimate_scope: EstimateScopeEnum
    target_regions: List[str] = Field(default=[])
    cohort_effect: Optional[float] = None


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
    # Scope of ate_estimate (#2053); see SimulationResponse.estimate_scope.
    estimate_scope: EstimateScopeEnum
    target_regions: List[str] = Field(default=[])
    filter_regions: List[str] = Field(default=[])


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
    # --- #2206 honest surfacing (all derived from the stored rows) ---
    fidelity_status: FidelityStatusEnum = Field(
        description=(
            "'unvalidated' while fidelity_score is NULL / fidelity_sample_count is 0 — "
            "no experiment outcome has been compared against this model; not a pass."
        ),
    )
    fidelity_score: Optional[float] = Field(
        default=None, description="Mean fidelity over the model's A/B comparisons; NULL = none."
    )
    fidelity_sample_count: int = Field(
        default=0, description="Number of experiment comparisons behind fidelity_score."
    )
    data_provenance: Optional[str] = Field(
        default=None,
        description="Training-frame provenance as recorded: 'synthetic' or 'rwd_file'.",
    )
    r2_score_basis: R2ScoreBasisEnum = Field(
        description=(
            "What r2_score was scored against. 'synthetic_target': the target is "
            "self-generated by the synthetic training frame, so R² measures how well the "
            "model reproduces its own generator — not real-world outcomes."
        ),
    )
    brand_is_feature: bool = Field(
        description="Whether 'brand' is one of the model's feature_columns (False: brand is routing metadata).",
    )
    training_fingerprint: str = Field(
        description=(
            "Content hash of the RECORDED fit: training_config (with the training_frame "
            "source/seed/rows when the trainer recorded it), feature/target columns, and "
            "every reported metric except wall-clock. Equal fingerprints = the same "
            "recorded fit under several labels; the artifact itself is not hashed."
        ),
    )
    shared_fit_model_count: int = Field(
        description=(
            "Distinct BRANDS whose active model of this twin_type has the same "
            "training_fingerprint, including this one (two active rows of one brand "
            "count once). >1 means brand is a label over ONE recorded fit."
        ),
    )
    shared_fit_with: List[str] = Field(
        default_factory=list,
        description="Other brands sharing this exact fit that the caller may read (brand-scoped).",
    )
    training_frame_recorded: bool = Field(
        description=(
            "Whether a CONTENT digest of the training frame "
            "(training_config.training_frame.content_sha256) was recorded for this row. "
            "False for rows trained before it was recorded: their fingerprint compares "
            "configuration, columns and metrics only — training-frame and artifact "
            "identity were not recorded."
        ),
    )


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
            models_available=0,
            brands_simulable=0,
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
            models_available=models_available,
            brands_simulable=0,
            simulations_pending=0,
            last_simulation_at=None,
        )

    model_brands, brands_simulable = await simulable_brands(repo.client, models)
    dark = bool(model_brands) and brands_simulable == 0

    return DigitalTwinHealthResponse(
        status="degraded" if dark else "healthy",
        service="digital-twin",
        models_available=models_available,
        brands_simulable=brands_simulable,
        simulations_pending=pending,
        last_simulation_at=last_simulation_at,
    )


# =============================================================================
# INTERVENTION TAXONOMY ENDPOINT (single source of truth for the dropdown)
# =============================================================================


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

    Availability is **brand-aware** and **per-intervention**. ``available`` is True
    only when a trained twin model exists for the brand/twin_type (otherwise
    ``/simulate`` would 503). ``available_for_effect`` is True only when THAT
    intervention's planted treatment channel has enough usable cohort rows for a
    direct causal estimate — honest per channel, so a substrate holding only some
    channels advertises exactly what ``/simulate`` can estimate. ``effect_basis`` is
    ``"cohort_causal"`` for identified interventions (direct DML estimate on the cohort)
    and ``"unavailable"`` otherwise (no fabricated effect; ``/simulate`` returns 422).
    """
    from src.digital_twin.effect.cohort_loader import cohort_treatment_availability
    from src.digital_twin.effect.provider import INTERVENTION_CATALOG
    from src.digital_twin.models.twin_models import TwinType

    available, effect_available = False, cast(Dict[str, bool], {})
    # An empty answer must say whether it is a FINDING or a lookup that did not complete (r3).
    resolution: Literal["resolved", "unavailable", "not_requested"] = "not_requested"
    effect_status: Optional[Literal["measured", "unmeasured"]] = None
    if brand is not None:
        try:
            repo = await _get_twin_repo()
            actives = await repo.list_active_models(
                twin_type=TwinType(twin_type.value), brand=brand.value
            )
            available, resolution, effect_status = len(actives) > 0, "resolved", "measured"
            # Per-intervention: an intervention is effect-available only when ITS planted
            # treatment channel has enough usable cohort rows (never all-or-nothing).
            effect_available = await cohort_treatment_availability(repo.client, brand.value)
            if available and not any(effect_available.values()):
                warn_model_without_effect_data(brand.value, effect_available)
                if effect_data_unmeasured(effect_available):
                    effect_status = "unmeasured"
        except Exception as e:  # repo/DB unreachable — degrade, never fabricate
            logger.warning("intervention-types: availability/cohort check failed: %s", e)
            available, effect_available, resolution, effect_status = False, {}, "unavailable", None

    items = [
        InterventionTypeItem(
            value=value,
            label=label,
            effect_basis=("cohort_causal" if effect_available.get(value, False) else "unavailable"),
            available=available,
            available_for_effect=effect_available.get(value, False),
        )
        for value, label in INTERVENTION_CATALOG
    ]
    return InterventionTypesResponse(
        interventions=items,
        model_resolution=resolution,
        effect_availability_status=effect_status,
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

        # The experiment link (#2206, codex r1 #1): the request always accepted
        # experiment_design_id and the save dropped it, so no twin_simulations row
        # was ever linked and the post-experiment fidelity producer — which resolves
        # the simulation experiment-scoped — could only skip. Validate it up front
        # (before the heavy work) and write it after the save.
        experiment_link: Optional[UUID] = None
        if request.experiment_design_id:
            try:
                experiment_link = UUID(str(request.experiment_design_id))
            except ValueError as bad_link:
                raise HTTPException(
                    status_code=422,
                    detail="experiment_design_id must be a UUID (the experiment this pre-screen is for).",
                ) from bad_link

        # Resolve a REAL trained model BEFORE generating: an explicit model_id, or
        # the highest-fidelity active model for this brand/twin_type. No model →
        # honest 503 (not a fresh untrained generator → opaque 500, and not a
        # UUID(int=0) sentinel → twin_simulations.model_id FK violation) (#705 H4).
        repo = await _get_twin_repo()
        if experiment_link is not None:
            # A well-formed id of a nonexistent or other-brand experiment would
            # leave a permanently orphaned pre-screen (codex r2 #1): verify first.
            await _verify_experiment_link(repo, experiment_link, request.brand.value)
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
            # Offload stays refused: the worker's cohort path (#2025) is uncertified live.
            raise HTTPException(
                status_code=503,
                detail=(
                    "Cohort-causal effect estimation runs inline only; the heavy-offload "
                    "path is not yet wired for it (set HEAVY_OFFLOAD_ENABLED=false)."
                ),
            )
        else:
            # P1 inline path (default + fallback). Twin generation + simulation are the
            # heavy, blocking, ~1.3 GiB part of this request: one in-flight heavy op per
            # worker (OOM guard), run off the event loop. When the per-worker slot budget
            # is exhausted heavy_compute_slot() raises HeavyComputeSaturated on enter
            # (mapped to a 503 + Retry-After by the app exception handler); nothing queues.
            generator = await _load_trained_generator(
                twin_type=twin_type, brand=brand, model_row=model_row
            )

            # Direction 2: estimate the effect DIRECTLY on the brand's cohort via a DML causal
            # estimate over the raw cohort frame from the cohort_provider built above the
            # offload/inline split; honest DML CI, no synthetic injected-effect handoff.
            from src.digital_twin.effect.cohort_causal_estimator import (
                CohortCausalEstimator,
            )

            # A region filter subsets the TWINS; scope the estimator to the same regions so
            # this surface and the chat simulator state the same ATE, CI, SE, recommendation
            # and sample size (#2023). The subgroup heterogeneity reports only the axes the
            # estimate resolves (#2054); confidence follows its training rows (#2104).
            target_regions = list(pop_filter.regions) if pop_filter else []

            def _do_sim():
                population = generator.generate(n=request.twin_count)
                engine = SimulationEngine(
                    population=population,
                    effect_provider=cohort_provider,
                    effect_estimator=CohortCausalEstimator(target_regions=target_regions),
                    # The model's measured fidelity (NULL until an experiment outcome
                    # has been compared against it). Never passed before, so the
                    # engine's gate could not fire even with a real score (#2206).
                    model_fidelity_score=model_row.get("fidelity_score"),
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

        # Save simulation result (reuse the repo resolved above), then link it to
        # its experiment so fidelity_tracking_update can find it (#2206).
        saved_id = await repo.save_simulation(result, request.brand.value)
        if experiment_link is not None:
            sim_id = saved_id or result.simulation_id
            linked = await repo.simulations.link_experiment(sim_id, experiment_link)
            if not linked:
                # The caller asked for a linked pre-screen and did not get one; a
                # 200 here would hide an orphan the fidelity loop can never find.
                raise HTTPException(
                    status_code=500,
                    detail=(
                        f"Simulation {sim_id} was saved but could not be linked to "
                        f"experiment {experiment_link}; link it before relying on "
                        "post-experiment fidelity tracking."
                    ),
                )

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
            fidelity_status=FidelityStatusEnum(result.fidelity_status.value),
            status=SimulationStatusEnum(result.status.value),
            error_message=result.error_message,
            execution_time_ms=result.execution_time_ms,
            is_significant=result.is_significant(),
            effect_direction=result.effect_direction(),
            created_at=result.created_at,
            data_provenance=result.data_provenance,
            estimate_scope=_live_estimate_scope(result.target_regions),
            target_regions=result.target_regions,
            cohort_effect=(None if result.cohort_ate is None else round(result.cohort_ate, 4)),
            cohort_ci_lower=(
                None if result.cohort_ci_lower is None else round(result.cohort_ci_lower, 4)
            ),
            cohort_ci_upper=(
                None if result.cohort_ci_upper is None else round(result.cohort_ci_upper, 4)
            ),
            effect_heterogeneity=_heterogeneity_response(result.effect_heterogeneity),
            subgroups_basis=_live_subgroups_basis(
                result.data_provenance, calculated=request.calculate_heterogeneity
            ),
        )

    except HTTPException:
        # 408 timeout (offload path) must propagate unchanged, not be swallowed
        # into a 500 by the broad handler below.
        raise
    except HeavyComputeSaturated:
        # Reject fast under load — surfaced as 503 + Retry-After by the app exception handler.
        # Must precede the broad handlers so it is not swallowed into a 500.
        raise
    except ValueError as e:
        raise rejected_request(logger, "simulation", "request", e)
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

        items = []
        for sim in paginated:
            scope = _stored_estimate_scope(sim)
            items.append(
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
                    estimate_scope=EstimateScopeEnum(scope.scope.value),
                    target_regions=scope.target_regions,
                    cohort_effect=_round4(scope.cohort_ate),
                )
            )

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
    brand: Optional[BrandEnum] = Query(None, description="Filter by brand (omit for all brands)"),
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

    # Fail-closed brand scoping (H11): an optional brand filter (None = all
    # brands the caller may see). A non-admin is still pinned to their grant;
    # admin / ['all'] sees all brands, or the one selected here.
    allowed, effective_brand = resolve_brand_for_read(user, brand.value if brand else None)
    if not allowed:
        raise HTTPException(status_code=403, detail="No brand grant for this user.")

    try:
        from src.digital_twin.twin_repository import stored_filter_regions

        repo = await _get_twin_repo()
        # Fetch enough rows to cover the requested window (repo returns newest
        # first), then apply the offset/limit slice.
        rows = await repo.simulations.list_simulations(brand=effective_brand, limit=offset + limit)
        window = rows[offset : offset + limit]

        items = []
        for sim in window:
            scope = _stored_estimate_scope(sim)
            items.append(
                SimulationHistoryItem(
                    simulation_id=str(sim.get("simulation_id", "")),
                    created_at=sim.get("created_at", datetime.now(timezone.utc)),
                    intervention_type=sim.get("intervention_type", "unknown"),
                    brand=sim.get("brand", "unknown"),
                    ate_estimate=round(sim.get("simulated_ate", 0.0), 4),
                    recommendation_type=sim.get("recommendation", "refine"),
                    data_provenance=sim.get("data_provenance"),
                    estimate_scope=EstimateScopeEnum(scope.scope.value),
                    target_regions=scope.target_regions,
                    filter_regions=stored_filter_regions(sim),
                )
            )

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
    from src.digital_twin.effect.cohort_causal_estimator import CohortCausalEstimator
    from src.digital_twin.effect.cohort_loader import build_cohort_provider_or_none
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
        # Direction 2 identification gate (same as /simulate): estimate a real causal
        # effect only for interventions identified in the cohort. A scenario whose
        # intervention is not identified fails the comparison closed (422) — never a
        # fabricated synthetic effect.
        cohort_provider = await build_cohort_provider_or_none(
            repo.client, scenario.intervention_type, scenario.brand
        )
        if cohort_provider is None:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"No effect data available for scenario intervention "
                    f"'{scenario.intervention_type}' / '{scenario.brand}': not identified "
                    "in the connected cohort (no fabricated effect is returned)."
                ),
            )
        generator = await _load_trained_generator(
            twin_type=twin_type, brand=brand, model_row=model_row
        )
        return generator, model_row, cohort_provider

    def _run_scenario(
        scenario: ScenarioSimulateRequest,
        generator: Any,
        model_row: Dict[str, Any],
        cohort_provider: Any,
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
        # Direction 2: real DML cohort estimate per scenario (no fabricated synthetic effect).
        engine = SimulationEngine(
            population=population,
            effect_provider=cohort_provider,
            effect_estimator=CohortCausalEstimator(),
            model_fidelity_score=model_row.get("fidelity_score"),  # #2206
        )
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
            fidelity_status=FidelityStatusEnum(result.fidelity_status.value),
            status=SimulationStatusEnum(result.status.value),
            error_message=result.error_message,
            execution_time_ms=result.execution_time_ms,
            is_significant=result.is_significant(),
            effect_direction=result.effect_direction(),
            created_at=result.created_at,
            data_provenance=result.data_provenance,
            estimate_scope=_live_estimate_scope(result.target_regions),
            effect_heterogeneity=_heterogeneity_response(result.effect_heterogeneity),
            subgroups_basis=_live_subgroups_basis(result.data_provenance),
        )

    try:
        repo = await _get_twin_repo()
        base_gen, base_row, base_provider = await _load_for(request.base_scenario)
        alt_loaded = [await _load_for(s) for s in request.alternative_scenarios]

        # Twin generation is the heavy, blocking, ~1.3 GiB work. Run every scenario
        # off the event loop under ONE per-worker heavy-compute slot (mirrors the
        # /simulate inline path) so a multi-scenario compare can't stall the worker
        # or bypass the OOM budget the slot enforces.
        async with heavy_compute_slot():
            base_result = await run_in_bounded_executor(
                _run_scenario, request.base_scenario, base_gen, base_row, base_provider
            )
            alternative_results = [
                await run_in_bounded_executor(_run_scenario, s, gen, row, prov)
                for s, (gen, row, prov) in zip(
                    request.alternative_scenarios, alt_loaded, strict=True
                )
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
    except ValueError as e:  # e.g. ``Brand(scenario.brand)``: "'X' is not a valid Brand"
        raise rejected_request(logger, "scenario comparison", "scenario", e)
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
        # SimulationResult (#705 H5b/H11). Map from it; derive what the row does not persist.
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

        from src.digital_twin.twin_repository import StoredSubgroupsBasis  # legacy rows, #2104

        scope = _stored_estimate_scope(result)
        eh = result.get("effect_heterogeneity") or {}
        heterogeneity = _heterogeneity_response(eh)

        # twin_simulations does not persist the model's fidelity; derive the explicit
        # state (and the warning) from the model row this run points at, as it
        # stands now (#2206). A missing model row validates nothing → unvalidated.
        model_row = None
        if result.get("model_id"):
            try:
                model_row = await repo.get_model(UUID(str(result["model_id"])))
            except Exception as model_err:  # pragma: no cover - defensive
                logger.warning("Model lookup failed for stored simulation: %s", model_err)
        fidelity_fields = _stored_fidelity_fields(model_row)

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
            **fidelity_fields,  # fidelity_status/_warning/_reason + model_fidelity_score
            status=SimulationStatusEnum(result.get("simulation_status", "completed")),
            error_message=result.get("error_message"),
            execution_time_ms=result.get("execution_time_ms", 0),
            is_significant=is_significant,
            effect_direction=effect_direction,
            created_at=result.get("created_at", datetime.now(timezone.utc)),
            completed_at=result.get("completed_at"),
            population_filters=result.get("population_filters") or {},
            effect_heterogeneity=heterogeneity,
            subgroups_basis=StoredSubgroupsBasis.from_row(result).value,
            intervention_config=result.get("intervention_config") or {},
            data_provenance=result.get("data_provenance"),  # #705 H5b
            estimate_scope=EstimateScopeEnum(scope.scope.value),  # #2053
            target_regions=scope.target_regions,
            cohort_effect=_round4(scope.cohort_ate),
            cohort_ci_lower=_round4(scope.cohort_ci_lower),
            cohort_ci_upper=_round4(scope.cohort_ci_upper),
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

        # Census over ALL brands first (shared-fit fingerprints, #2206), then the
        # brand-scoped listing is a filter over it.
        census = await _active_model_census(repo, twin_type_enum)
        models = [
            m for m in census if effective_brand is None or str(m.get("brand")) == effective_brand
        ]

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
                    **_model_honesty_fields(m, census, user),
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
        from src.digital_twin.models.twin_models import TwinType

        census = await _active_model_census(
            repo, TwinType(model["twin_type"]) if model.get("twin_type") else None
        )
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
            **_model_honesty_fields(model, census, user),
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
