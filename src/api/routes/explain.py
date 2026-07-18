"""
E2I Real-Time Model Interpretability API
=========================================
FastAPI endpoint for real-time SHAP explanations alongside predictions.

Pattern inspired by: https://medium.com/towards-data-science/real-time-model-interpretability-api-using-shap-streamlit-and-docker-e664d9797a9a

Integration Points:
- BentoML model serving (prediction)
- SHAP explainer (real-time local explanations)
- ml_shap_analyses table (audit trail)
- prediction_synthesizer agent (downstream consumer)

Author: E2I Causal Analytics Team
Version: 4.2.0
"""

import asyncio
import inspect
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from src.api.dependencies.auth import require_auth

# Real implementations
from src.api.dependencies.bentoml_client import BentoMLClient, get_bentoml_client
from src.api.schemas.errors import ErrorResponse, ValidationErrorResponse
from src.api.utils.data_masking import mask_identifier
from src.feature_store.feast_client import FeastClient, get_feast_client
from src.feature_store.online_feature_presence import missing_or_null_feature_fields
from src.mlops.shap_explainer_realtime import RealTimeSHAPExplainer, SHAPResult
from src.repositories.shap_analysis import ShapAnalysisRepository, get_shap_analysis_repository

logger = logging.getLogger(__name__)

# P2 offload: max seconds the synchronous SHAP endpoints wait for the
# worker_heavy task before returning 408. Kept just under the frontend's 30s
# axios timeout (frontend/src/lib/api-client.ts) so the client gets a clean 408.
# Only consulted when HEAVY_OFFLOAD_ENABLED is set; the inline path is unaffected.
_SHAP_OFFLOAD_TIMEOUT_SECONDS = 28.0

router = APIRouter(
    prefix="/explain",
    tags=["Model Interpretability"],
    responses={
        401: {"model": ErrorResponse, "description": "Authentication required"},
        422: {"model": ValidationErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)


# =============================================================================
# ENUMS & MODELS
# =============================================================================


class ModelType(str, Enum):
    """Supported model types for SHAP explanation.

    The legacy four (propensity / risk_stratification / next_best_action /
    churn_prediction) back the original demonstration taxonomy. The three
    gold-standard cohort families (#39) back the REAL deployed
    ``*_goldstd_lr_v1`` registry models: they serve the 3 RAW leakage-safe
    KEEP_COLUMNS covariates and SHAP runs over the 9 ENCODED features produced
    by the bundled FeatureBuilder.
    """

    PROPENSITY = "propensity"
    RISK_STRATIFICATION = "risk_stratification"
    NEXT_BEST_ACTION = "next_best_action"
    CHURN_PREDICTION = "churn_prediction"
    # Gold-standard cohort families (#39) — real deployed per-brand models.
    # Patient-grain (3 raw covariates → 9 encoded):
    INITIATION = "initiation"
    PERSISTENCE = "persistence"
    DISCONTINUATION = "discontinuation"
    # HCP-grain (5 raw covariates → 19 encoded; JOIN-embedded from hcp_profiles):
    HCP_ADOPTION = "hcp_adoption"


# Gold-standard cohort families: served by the real per-brand ``*_goldstd_lr_v1``
# models, whose serving contract is RAW covariates + a bundled FeatureBuilder
# (SHAP over the ENCODED vector). Used to branch the feature-resolution / numeric
# guard and to resolve the per-brand serving model_name.
GOLDSTD_COHORT_MODEL_TYPES: frozenset[ModelType] = frozenset(
    {
        ModelType.INITIATION,
        ModelType.PERSISTENCE,
        ModelType.DISCONTINUATION,
        ModelType.HCP_ADOPTION,
    }
)

# The brands the gold-standard models are registered for (mirrors
# src/mlops/gold_standard_eval/cohort_spec.BRANDS). Lower-cased into the serving
# name. Kept local so this route module does not import the heavy mlops package.
GOLDSTD_BRANDS: tuple[str, ...] = ("Remibrutinib", "Fabhalta", "Kisqali")
_DEFAULT_GOLDSTD_BRAND = "Remibrutinib"

# Adaptive-sizing cap for /global, per model type. Persistence/discontinuation
# top-5 rankings carry mid-tail pairs whose real gaps (~0.01-0.06 mean |SHAP|)
# are still noisy at n=60 and need n≈150-200 to separate (measured from stored
# sample points, 2026-07-18), so those cohorts default to the full ceiling; the
# other cohorts stabilize by n≈25-60 and keep the cheaper default. ~200
# sequential SHAP calls run ≈1 min, well inside the 300 s /api proxy budget.
_MAX_SAMPLE_SIZE_CEILING = 200
_DEFAULT_MAX_SAMPLE_SIZE = 60
_EXTENDED_CAP_MODEL_TYPES: frozenset[ModelType] = frozenset(
    {ModelType.PERSISTENCE, ModelType.DISCONTINUATION}
)


def _default_max_sample_size(model_type: ModelType) -> int:
    """Default adaptive-sizing cap for a cohort (see _EXTENDED_CAP_MODEL_TYPES)."""
    if model_type in _EXTENDED_CAP_MODEL_TYPES:
        return _MAX_SAMPLE_SIZE_CEILING
    return _DEFAULT_MAX_SAMPLE_SIZE


def goldstd_serving_name(model_type: ModelType, brand: Optional[str]) -> str:
    """Resolve the per-brand serving ``model_name`` for a gold-standard cohort.

    Returns ``f"{cohort}_{brand}_goldstd_lr_v1"`` (e.g.
    ``initiation_remibrutinib_goldstd_lr_v1``,
    ``hcp_adoption_kisqali_goldstd_lr_v1``) — the registry model_name the
    multi-model BentoML service routes on. ``brand`` is case-insensitive; when
    None it defaults to the explicit default brand (NOT a silent skip). An
    unknown brand FAILS CLOSED (422) so no wrong-model prediction is produced.
    """
    chosen = brand or _DEFAULT_GOLDSTD_BRAND
    match = next((b for b in GOLDSTD_BRANDS if b.lower() == chosen.lower()), None)
    if match is None:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unknown brand '{brand}' for gold-standard cohort "
                f"'{model_type.value}'; valid brands: {list(GOLDSTD_BRANDS)}"
            ),
        )
    return f"{model_type.value}_{match.lower()}_goldstd_lr_v1"


class ExplanationFormat(str, Enum):
    """Output format for SHAP explanations."""

    FULL = "full"  # All SHAP values + metadata
    TOP_K = "top_k"  # Only top K contributing features
    NARRATIVE = "narrative"  # NL explanation (requires Claude)
    MINIMAL = "minimal"  # Prediction + top 3 features only


class FeatureContribution(BaseModel):
    """Single feature's contribution to prediction."""

    feature_name: str = Field(..., description="Name of the feature")
    feature_value: Any = Field(..., description="Actual value of feature for this instance")
    shap_value: float = Field(..., description="SHAP contribution to prediction")
    contribution_direction: str = Field(..., description="positive or negative")
    contribution_rank: int = Field(..., description="Rank by absolute SHAP value")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "feature_name": "days_since_last_hcp_visit",
                "feature_value": 45,
                "shap_value": 0.234,
                "contribution_direction": "positive",
                "contribution_rank": 1,
            }
        }
    )


class ExplainRequest(BaseModel):
    """Request payload for real-time explanation."""

    patient_id: str = Field(..., description="Patient identifier")
    hcp_id: Optional[str] = Field(None, description="HCP context for the prediction")
    model_type: ModelType = Field(..., description="Type of model to explain")
    brand: Optional[str] = Field(
        None,
        description=(
            "Brand for the gold-standard per-brand cohort models (#39): "
            "Remibrutinib | Fabhalta | Kisqali. Selects which per-brand "
            "model to explain (serving name f'{cohort}_{brand}_goldstd_lr_v1'). "
            "Defaults to Remibrutinib when omitted for a gold-standard cohort; "
            "ignored for the legacy model types."
        ),
    )
    model_version_id: Optional[str] = Field(
        None, description="Specific model version (latest if not specified)"
    )
    features: Optional[Dict[str, Any]] = Field(
        None, description="Pre-computed features (fetched from Feast if not provided)"
    )
    format: ExplanationFormat = Field(default=ExplanationFormat.TOP_K, description="Output format")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of top features to return")
    include_base_value: bool = Field(
        default=True, description="Include model's base prediction value"
    )
    store_for_audit: bool = Field(
        default=True, description="Store explanation in ml_shap_analyses for compliance"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "patient_id": "PAT-2024-001234",
                "hcp_id": "HCP-NE-5678",
                "model_type": "propensity",
                "format": "top_k",
                "top_k": 5,
                "store_for_audit": True,
            }
        }
    )


class ExplainResponse(BaseModel):
    """Response payload with prediction + SHAP explanation."""

    # Identifiers
    explanation_id: str = Field(..., description="Unique ID for this explanation (for audit trail)")
    request_timestamp: datetime = Field(..., description="When request was received")

    # Prediction
    patient_id: str
    model_type: ModelType
    model_version_id: str
    prediction_class: str = Field(..., description="Predicted class label")
    prediction_probability: float = Field(..., description="Prediction confidence [0-1]")

    # SHAP Explanation
    base_value: Optional[float] = Field(
        None, description="Model's expected value (average prediction)"
    )
    top_features: List[FeatureContribution] = Field(..., description="Top contributing features")
    shap_sum: float = Field(
        ..., description="Sum of all SHAP values (should equal prediction - base_value)"
    )

    # Optional narrative
    narrative_explanation: Optional[str] = Field(
        None, description="Natural language explanation (if format=narrative)"
    )

    # Metadata
    computation_time_ms: float = Field(..., description="Time to compute explanation")
    audit_stored: bool = Field(..., description="Whether explanation was stored for compliance")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "explanation_id": "EXPL-2024-abc123",
                "request_timestamp": "2024-12-15T10:30:00Z",
                "patient_id": "PAT-2024-001234",
                "model_type": "propensity",
                "model_version_id": "v2.3.1-prod",
                "prediction_class": "high_propensity",
                "prediction_probability": 0.78,
                "base_value": 0.42,
                "top_features": [
                    {
                        "feature_name": "days_since_last_hcp_visit",
                        "feature_value": 45,
                        "shap_value": 0.15,
                        "contribution_direction": "positive",
                        "contribution_rank": 1,
                    }
                ],
                "shap_sum": 0.36,
                "narrative_explanation": None,
                "computation_time_ms": 127.5,
                "audit_stored": True,
            }
        }
    )


class BatchExplainRequest(BaseModel):
    """Batch explanation request for multiple patients."""

    requests: List[ExplainRequest] = Field(
        ..., max_length=50, description="Up to 50 patients per batch"
    )
    parallel: bool = Field(default=True, description="Process in parallel")


class BatchExplainResponse(BaseModel):
    """Batch explanation response."""

    batch_id: str
    total_requests: int
    successful: int
    failed: int
    explanations: List[ExplainResponse]
    errors: List[Dict[str, str]]
    total_time_ms: float


class GlobalImportanceFeature(BaseModel):
    """One feature's SHAP importance aggregated over a cohort sample (#39 global view).

    ``mean_abs_shap`` is the canonical *global feature importance* (the standard
    SHAP global ranking statistic — mean |SHAP| over a sample). ``mean_shap`` is
    the net signed direction across the sample.
    """

    feature_name: str = Field(..., description="Encoded feature name")
    mean_abs_shap: float = Field(
        ..., description="Mean |SHAP| across the sample (global importance ranking)"
    )
    mean_shap: float = Field(..., description="Mean signed SHAP across the sample (net direction)")
    mean_feature_value: Optional[float] = Field(
        None, description="Mean feature value across the sample (numeric features only)"
    )
    contribution_rank: int = Field(..., description="Rank by mean_abs_shap (1 = most important)")


class GlobalImportancePoint(BaseModel):
    """One entity's signed SHAP for one feature — a REAL beeswarm dot.

    The points come from the per-entity SHAP that backed the aggregation, so the
    beeswarm shows a genuine distribution (not a single fabricated dot per feature).
    """

    feature_name: str
    shap_value: float
    feature_value: Optional[float] = None


class GlobalFeatureImportanceResponse(BaseModel):
    """Cohort-level (global) SHAP feature importance for one per-brand model (#39)."""

    model_type: ModelType
    brand: str
    model_name: str = Field(
        ..., description="Resolved serving name, e.g. initiation_kisqali_goldstd_lr_v1"
    )
    base_value: Optional[float] = Field(None, description="Mean model base value across the sample")
    sample_size: int = Field(
        ..., description="Entities successfully explained (n_succeeded, honest)"
    )
    requested_sample_size: int = Field(
        ..., description="Minimum sample size requested (adaptive sizing may exceed it)"
    )
    computation_method: str = Field(..., description="SHAP explainer used (e.g. LinearExplainer)")
    computed_at: datetime
    cached: bool = Field(..., description="True when read from a stored precomputed row")
    sampling_method: Optional[str] = Field(
        None,
        description=(
            "'random' (uniform draw via RPC) or 'prefix_fallback' (deterministic, "
            "pre-migration DB); None on legacy cached rows"
        ),
    )
    stability_achieved: Optional[bool] = Field(
        None,
        description=(
            "True when the top-feature mean-|SHAP| ranking separated beyond "
            "sampling noise at the final n; None on legacy cached rows"
        ),
    )
    stopping_reason: Optional[str] = Field(
        None,
        description="stable | max_sample_size_reached | candidates_exhausted",
    )
    features: List[GlobalImportanceFeature] = Field(
        ..., description="Features ranked desc by mean_abs_shap"
    )
    points: List[GlobalImportancePoint] = Field(
        default_factory=list,
        description="Per-entity SHAP points for the top features (real beeswarm distribution)",
    )


class SampleEntitiesResponse(BaseModel):
    """Real entity IDs for the per-entity SHAP picker (#39)."""

    model_type: ModelType
    grain: str = Field(..., description="'patient' or 'hcp'")
    id_field: str = Field(..., description="patient_id | hcp_id")
    entities: List[str] = Field(..., description="Real entity identifiers from the cohort source")


# =============================================================================
# DEPENDENCY INJECTION
# =============================================================================


# =============================================================================
# HELPERS (Issue #321 — FE/BE contract drift cleanup)
# =============================================================================


def _normalize_history_row(row: Dict[str, Any], masked_patient_id: str) -> Dict[str, Any]:
    """Normalize an ``ml_shap_analyses`` row to the FE ``ExplainResponse`` shape.

    The FE type ``ExplanationHistoryResponse`` (see
    ``frontend/src/types/explain.ts``) declares ``explanations: ExplainResponse[]``.
    The DB row uses different column names and stores SHAP values as a dict;
    this helper bridges the two without forcing a Pydantic re-validation
    (which would fail on missing optional fields and reject legacy rows).

    Column-name precedence for the SHAP map (signed per-feature contributions):

    1. ``local_shap_values`` — canonical column in ``ml_shap_analyses`` per
       ``database/ml/mlops_tables.sql``. Local-realtime rows store signed
       per-patient SHAP contributions here.
    2. ``shap_values`` — referenced by views/functions in migration
       ``database/ml/011_realtime_shap_audit.sql`` for legacy callers.
    3. ``global_importance`` — last-resort fallback. NOTE: this column
       stores mean-absolute importance, so sign is lost. Only used when
       neither of the signed columns is present (legacy global rows).
    """
    shap_values: Dict[str, Any] = (
        row.get("local_shap_values") or row.get("shap_values") or row.get("global_importance") or {}
    )
    if isinstance(shap_values, dict):
        ranked = sorted(
            shap_values.items(),
            key=lambda kv: abs(float(kv[1]) if kv[1] is not None else 0.0),
            reverse=True,
        )
        top_features = [
            {
                "feature_name": name,
                "feature_value": None,
                "shap_value": float(value) if value is not None else 0.0,
                "contribution_direction": "positive"
                if (value is not None and float(value) >= 0)
                else "negative",
                "contribution_rank": idx + 1,
            }
            for idx, (name, value) in enumerate(ranked)
        ]
        shap_sum = float(sum(float(v) for v in shap_values.values() if v is not None))
    else:
        top_features = []
        shap_sum = 0.0

    request_ts = row.get("request_timestamp") or row.get("computed_at")

    # NOTE: ``model_type`` and ``model_version_id`` are NOT columns on the
    # current ``ml_shap_analyses`` schema (only ``model_registry_id`` FK is).
    # The FE type declares them as required strings, so we surface ``""``
    # when the row carries nothing — better than null which would break
    # FE deserialization. A follow-up issue tracks adding these as proper
    # columns + persisting them in the audit-write path.
    return {
        "explanation_id": row.get("explanation_id") or row.get("id") or "",
        "request_timestamp": request_ts,
        "patient_id": masked_patient_id,
        "model_type": row.get("model_type") or "",
        "model_version_id": row.get("model_version_id") or row.get("model_registry_id") or "",
        "prediction_class": row.get("prediction_class") or "",
        "prediction_probability": float(row.get("prediction_probability") or 0.0),
        "base_value": (float(row["base_value"]) if row.get("base_value") is not None else None),
        "top_features": top_features,
        "shap_sum": shap_sum,
        "narrative_explanation": row.get("natural_language_explanation"),
        "computation_time_ms": float(row.get("response_time_ms") or 0.0),
        "audit_stored": True,
    }


def _get_user_hcps(user: Dict[str, Any]) -> List[str]:
    """Extract the caller's HCP-assignment grants from the user dict.

    The ``ml_shap_analyses`` rows are scoped by ``hcp_id`` — the documented
    object-level authorization model (``database/ml/011_realtime_shap_audit.sql``
    RLS): a field rep may only see explanations whose ``hcp_id`` is in their
    assignment set; admins / data scientists see all.

    Look-up order mirrors ``src.api.dependencies.auth.get_user_brands`` (which
    reads ``app_metadata.brands``), applied to HCP grants:

    1. ``app_metadata.hcps`` (Supabase convention)
    2. top-level ``hcps`` field
    3. Empty list when neither is set

    ``['all']`` means cross-scope access (treated like an admin grant).
    """
    hcps = user.get("app_metadata", {}).get("hcps")
    if hcps is None:
        hcps = user.get("hcps", [])
    if isinstance(hcps, str):
        return [hcps]
    return list(hcps or [])


def _caller_can_view_row(user: Dict[str, Any], row: Dict[str, Any]) -> bool:
    """Object-level authorization for a single ``ml_shap_analyses`` row.

    Returns True when the caller is allowed to view this explanation:

    * Admins (role ADMIN) — cross-scope, mirrors the RLS ``admin_access`` policy.
    * Callers carrying the cross-scope ``'all'`` HCP grant.
    * Otherwise the row's ``hcp_id`` must be in the caller's HCP grant set.

    A row with no ``hcp_id`` is only visible to admins / ``'all'`` callers — a
    scoped caller has no grant that could match it, so it stays hidden
    (fail-closed; an explanation must not leak to a rep who can't be tied to it).
    """
    from src.api.dependencies.auth import UserRole, has_role

    allowed_hcps = set(_get_user_hcps(user))
    if has_role(user, UserRole.ADMIN) or "all" in allowed_hcps:
        return True
    row_hcp = row.get("hcp_id")
    return bool(row_hcp) and row_hcp in allowed_hcps


async def _get_latest_versions_by_model_type() -> Dict[str, Optional[str]]:
    """Look up the latest ``model_version`` for each ``model_name`` in the
    registry, keyed by the matching ``ModelType.value``.

    Returns ``{model_type_value: latest_version_or_None}`` for every member
    of :class:`ModelType`. Best-effort: any error yields all-``None`` values
    so the response shape stays stable even with no DB connection.
    """
    versions: Dict[str, Optional[str]] = {mt.value: None for mt in ModelType}
    try:
        from src.memory.services.factories import get_async_supabase_client

        client = await get_async_supabase_client()
        if client is None:
            return versions

        # Fetch (model_name, model_version) for any of our model types. The
        # set is small (4); a single SELECT with ``in_`` is cheaper than 4
        # per-type round-trips. ``ml_model_registry.registered_at`` is the
        # canonical timestamp column (see database/ml/mlops_tables.sql:166);
        # there is no ``created_at`` column on this table.
        # Provenance (#894): ml_model_registry is is_synthetic-tagged
        # (migration 069; 720/722 live rows synthetic) — without the predicate
        # a synthetic row wins the latest-version race on this user route.
        result = await (
            client.table("ml_model_registry")
            .select("model_name,model_version,registered_at")
            .in_("model_name", list(versions.keys()))
            .eq("is_synthetic", False)
            .order("registered_at", desc=True)
            .execute()
        )

        rows: List[Dict[str, Any]] = result.data or []
        # Order desc by registered_at — first row per model_name wins.
        for r in rows:
            name = r.get("model_name")
            if name in versions and versions[name] is None:
                versions[name] = r.get("model_version")

        # Gold-standard cohorts (#39): their registry rows are named
        # ``<cohort>_<brand>_goldstd_lr_v1`` / ``csu_initiation_goldstd_lr_v1`` /
        # ``pnh_persistence_goldstd_lr_v1`` — NOT the bare ``initiation`` etc. So
        # the ``in_`` above cannot match them. Resolve the cohort-family latest
        # version with a second LIKE query and map each row to its family. These
        # are real fitted models registered at stage='staging', is_synthetic=
        # False, so the same is_synthetic=False predicate applies.
        goldstd = await (
            client.table("ml_model_registry")
            .select("model_name,model_version,registered_at")
            .like("model_name", "%_goldstd_lr_v1")
            .eq("is_synthetic", False)
            .order("registered_at", desc=True)
            .execute()
        )
        for r in goldstd.data or []:
            family = _goldstd_cohort_family(r.get("model_name") or "")
            if family in versions and versions[family] is None:
                versions[family] = r.get("model_version")

        return versions

    except Exception as e:
        # Surface as debug — this is best-effort enrichment, not load-bearing.
        logger.debug(f"latest_version enrichment unavailable: {e}")
        return versions


def _goldstd_cohort_family(model_name: str) -> Optional[str]:
    """Map a ``*_goldstd_lr_v1`` registry model_name to its cohort FAMILY.

    The registry carries per-brand (``initiation_remibrutinib_goldstd_lr_v1``)
    and base/aggregate (``csu_initiation_goldstd_lr_v1`` /
    ``pnh_persistence_goldstd_lr_v1`` / ``pnh_discontinuation_goldstd_lr_v1``)
    names; all collapse to one of the three ModelType cohort families. Returns
    ``None`` for a non-gold-standard name.
    """
    if not model_name.endswith("_goldstd_lr_v1"):
        return None
    lowered = model_name.lower()
    # ``hcp_adoption`` is the 4th cohort (HCP grain). It MUST be checked — its 3
    # ``hcp_adoption_{brand}_goldstd_lr_v1`` models are registered just like the
    # patient cohorts; omitting it made /explain/models report
    # ``latest_version: null`` for hcp_adoption despite live registry rows.
    for family in ("initiation", "persistence", "discontinuation", "hcp_adoption"):
        if family in lowered:
            return family
    return None


class RealTimeSHAPService:
    """
    Service layer for real-time SHAP explanations.

    This class orchestrates:
    1. Feature retrieval from Feast (if not provided)
    2. Prediction from BentoML endpoint
    3. SHAP computation (local explanations)
    4. Audit storage in ml_shap_analyses
    5. Optional narrative generation via Claude
    """

    def __init__(
        self,
        bentoml_client: Optional[BentoMLClient] = None,
        shap_explainer: Optional[RealTimeSHAPExplainer] = None,
        shap_repo: Optional[ShapAnalysisRepository] = None,
        feast_client: Optional[FeastClient] = None,
    ):
        """Initialize with real or injected dependencies."""
        self.bentoml_client = bentoml_client
        self.shap_explainer = shap_explainer or RealTimeSHAPExplainer()
        self.shap_repo = shap_repo
        self.feast_client = feast_client
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        """Lazy initialization of async dependencies."""
        if self._initialized:
            return

        # Initialize BentoML client if not provided
        if self.bentoml_client is None:
            try:
                self.bentoml_client = await get_bentoml_client()
            except Exception as e:
                logger.warning(f"BentoML client not available: {e}")

        # Initialize Feast client if not provided
        if self.feast_client is None:
            try:
                self.feast_client = await get_feast_client()
            except Exception as e:
                logger.warning(f"Feast client not available: {e}")

        # Initialize SHAP repository if not provided
        if self.shap_repo is None:
            try:
                self.shap_repo = await get_shap_analysis_repository()
            except Exception as e:
                logger.warning(f"SHAP repository not available: {e}")

        self._initialized = True

    async def get_features(
        self,
        patient_id: str,
        model_type: ModelType,
        entity_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Retrieve features from the Feast feature store.

        Fails LOUD when Feast is unavailable. This route feeds a regulatory-audit
        SHAP explanation, so it must NOT silently substitute fabricated default
        features (the #532 silent-degradation contract) — that would present
        invented patient data as a real, audit-grade explanation. Mirrors the
        predictions route, which returns 503 on a Feast lookup failure.

        Grain-aware entity (#39 multi-model): patient-grain cohorts key on
        ``patient_id``; the HCP-grain ``hcp_adoption`` cohort keys on ``hcp_id``
        (``entity_id`` carries the hcp_id, falling back to ``patient_id`` only
        when no explicit hcp entity is supplied).

        Raises:
            HTTPException: 503 when the Feast client is unavailable or the
                online-feature lookup fails.
        """
        await self._ensure_initialized()

        if self.feast_client is None:
            raise HTTPException(
                status_code=503,
                detail="Feature store unavailable: Feast client not initialized",
            )

        try:
            # Map model type to feature refs
            feature_refs = self._get_feature_refs_for_model(model_type)

            # HCP-grain cohort → key on hcp_id; patient cohorts → patient_id.
            if model_type == ModelType.HCP_ADOPTION:
                hcp_entity = entity_id or patient_id
                entity_row: Dict[str, Any] = {"hcp_id": hcp_entity}
            else:
                entity_row = {"patient_id": patient_id}

            features_dict = await self.feast_client.get_online_features(
                entity_rows=[entity_row],
                feature_refs=feature_refs,
                full_feature_names=False,
            )

            # Convert list values to single values (since we're querying one patient)
            features = {k: v[0] if v else None for k, v in features_dict.items()}

            # #576 anti-null-trap guard: a Feast 200 can carry PRESENT-but-null
            # values (verified live for a single-key lookup against a
            # composite-keyed patient view). This route feeds a regulatory-audit
            # SHAP record, so a null/incomplete vector must FAIL LOUD (503) — it
            # must NOT be explained and persisted as a real, audit-grade record
            # (#532/#576). Distinct from the feast-error path below: here the
            # lookup SUCCEEDS but returns nulls, which an exception guard misses.
            missing = missing_or_null_feature_fields(features, feature_refs)
            if missing:
                logger.error(
                    "Feast returned null/missing required features %s for "
                    "patient=%s (model=%s); refusing to build a SHAP/audit "
                    "record over an incomplete vector (#576).",
                    missing,
                    mask_identifier(patient_id),
                    model_type.value,
                )
                raise HTTPException(
                    status_code=503,
                    detail="Feature store returned incomplete features",
                )

            return features

        except HTTPException:
            raise
        except Exception as e:
            # Mask the patient_id in logs to match the PII-masking this route
            # enforces on responses (a Feast outage must not leak raw IDs to logs).
            logger.error(
                f"Feast feature retrieval failed for patient={mask_identifier(patient_id)}: {e}"
            )
            raise HTTPException(
                status_code=503,
                detail=f"Feature store lookup failed: {e}",
            ) from e

    def _get_feature_refs_for_model(self, model_type: ModelType) -> List[str]:
        """Get feature references for a model type.

        Delegates to the canonical registry in
        ``src/feature_store/model_feature_refs.py`` (3A-M-1). Note: the
        legacy contract here returned ``[]`` for unknown ``model_type``
        rather than the propensity fallback used by the predictions
        route — preserve that to avoid regressing existing call sites
        that explicitly check for empty list.
        """
        # Use raw value; ``feature_refs_for_model`` defaults to propensity
        # when the key is missing — we have to distinguish "unknown
        # model" from "fallback to propensity" because explain.py's
        # legacy semantics return []. Look up directly instead.
        from src.feature_store.model_feature_refs import MODEL_FEATURE_REFS

        return list(MODEL_FEATURE_REFS.get(model_type.value, []))

    def _get_default_features(self) -> Dict[str, Any]:
        """Static default features for unit tests only.

        NOT a production fallback: ``get_features`` fails loud (HTTP 503) when
        Feast is unavailable rather than substituting these fabricated values
        into a real SHAP explanation / regulatory-audit record (#532).
        """
        return {
            "days_since_last_hcp_visit": 45,
            "total_hcp_interactions_90d": 12,
            "therapy_adherence_score": 0.72,
            "lab_value_trend": 0.15,
            "prior_brand_experience": 1,
            "insurance_tier": 2,
            "region": 1,
            "hcp_specialty_match": 1,
            "patient_age_bucket": 3,
            "comorbidity_count": 2,
        }

    async def resolve_canonical_model_features(
        self,
        features: Dict[str, Any],
        model_type: ModelType,
        serving_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Resolve the strictly-validated, model-ordered feature dict.

        Two contracts, branched on the served model's ``/model_info``:

        * **Bare-estimator / ColumnTransformer (legacy):** the caller must supply
          the model's pre-encoded numeric ``feature_columns``. Returns
          ``{feature_name: float}`` for exactly those columns, in order. FAILS
          CLOSED (422) if a required feature is missing, null, or non-numeric —
          the #532/#576 audit contract (no hash/zero-fill fabrication).

        * **FeatureBuilder gold-standard cohort (#39):** ``/model_info`` exposes
          ``keep_columns`` (the RAW covariate names) alongside the encoded
          ``feature_columns``. The caller supplies the RAW covariates; the served
          FeatureBuilder one-hot/median-encodes them into the numeric encoded
          vector SHAP runs over. Validation is therefore against ``keep_columns``,
          and the categorical RAW covariate (e.g. ``geographic_region``) is
          ALLOWED as a string (it would be float-coerce-rejected by the legacy
          guard). It STILL FAILS CLOSED (422) on a missing/null RAW covariate —
          no fabrication. Returns ``{raw_covariate: value}`` (values keep their
          native type so the categorical string survives to the encoder).

        Both contracts FAIL CLOSED (503) if the model exposes no feature order.
        Extra non-model fields in the request are IGNORED.
        """
        if not self.bentoml_client:
            raise HTTPException(
                status_code=503,
                detail="Model serving unavailable: BentoML client not configured",
            )
        # The serving name selects WHICH model the multi-model BentoML service
        # describes (#39). Defaults to model_type.value (legacy single-model).
        info_name = serving_name or model_type.value
        try:
            info = await self.bentoml_client.get_model_info(info_name)
        except Exception as e:
            logger.error("Could not fetch model_info for SHAP prediction: %s", e)
            raise HTTPException(
                status_code=503,
                detail=f"Model metadata unavailable for '{info_name}'",
            )

        # FeatureBuilder gold-standard branch (#39): validate against the RAW
        # covariate names and allow a categorical string for the column the
        # FeatureBuilder will one-hot-encode. The presence of a non-empty
        # ``keep_columns`` list is the signal that a preprocessor will encode the
        # raw covariates (so SHAP runs over the encoded numeric vector).
        keep_columns = info.get("keep_columns")
        if isinstance(keep_columns, list) and keep_columns:
            encoded_cols = info.get("feature_columns")
            encoded_list = encoded_cols if isinstance(encoded_cols, list) else []
            return self._resolve_raw_covariates(features, keep_columns, encoded_list, model_type)

        feature_order = info.get("feature_columns")
        if not feature_order or not isinstance(feature_order, list):
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Model '{model_type.value}' exposes no feature order; cannot "
                    "produce an audit-grade prediction for SHAP"
                ),
            )

        missing = [name for name in feature_order if name not in features or features[name] is None]
        if missing:
            raise HTTPException(
                status_code=422,
                detail=f"Missing/null required feature(s) for SHAP prediction: {missing}",
            )

        canonical: Dict[str, float] = {}
        for name in feature_order:
            raw = features[name]
            # Only genuine numerics/bools pass. Strings (which would be
            # hash-encoded) and objects (zero-filled) FAIL CLOSED.
            if not isinstance(raw, (int, float, bool)):
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"Required feature '{name}' must be numeric for an "
                        f"audit-grade SHAP prediction (got {type(raw).__name__})"
                    ),
                )
            canonical[name] = float(raw)
        return canonical

    @staticmethod
    def _infer_categorical_covariates(
        keep_columns: List[str], encoded_feature_columns: List[str]
    ) -> set[str]:
        """Infer which RAW covariates are categorical from the encoded columns.

        GRAIN-AGNOSTIC (no hardcoded name): the FeatureBuilder encodes a numeric
        covariate ``X`` as a bare ``X`` column; a categorical ``X`` becomes only
        one-hot ``X_<value>`` columns. So ``X`` is categorical iff the encoded
        set has NO bare ``X`` but DOES have an ``X_``-prefixed column. When the
        encoded columns are unavailable (legacy/empty), fall back to the
        historical patient categorical (``geographic_region``) so the patient
        path is unchanged.
        """
        encoded = set(encoded_feature_columns)
        if not encoded:
            return {"geographic_region"} & set(keep_columns)
        categorical: set[str] = set()
        for name in keep_columns:
            if name in encoded:
                continue  # bare encoded column → numeric covariate
            if any(c.startswith(f"{name}_") for c in encoded):
                categorical.add(name)
        return categorical

    def _resolve_raw_covariates(
        self,
        features: Dict[str, Any],
        keep_columns: List[str],
        encoded_feature_columns: List[str],
        model_type: ModelType,
    ) -> Dict[str, Any]:
        """Validate RAW covariates for a FeatureBuilder gold-standard model (#39).

        The served FeatureBuilder encodes these RAW covariates into the numeric
        vector SHAP runs over, so a categorical RAW covariate (e.g.
        ``geographic_region`` for patient cohorts, ``specialty`` +
        ``geographic_region`` for HCP) is ALLOWED as a string here — the strict
        all-numeric guard is for the legacy pre-encoded contract only.

        Which covariates are categorical is INFERRED from the model's own encoded
        ``feature_columns`` (GRAIN-AGNOSTIC — no hardcoded name): a numeric
        covariate ``X`` appears as bare ``X`` (+ ``X__isna``) in the encoded set;
        a categorical ``X`` appears only as one-hot ``X_<value>`` columns. So
        ``X`` is categorical iff there is NO bare ``X`` encoded column but there
        ARE ``X_``-prefixed one-hot columns.

        Still FAILS CLOSED (422) on a missing/null RAW covariate, and on a type
        mismatch (a string in a numeric covariate, a non-string in a categorical)
        — no fabricated value reaches an audit-grade SHAP record (#532/#576).

        Returns ``{raw_covariate: value}`` for exactly ``keep_columns``, values
        keeping their native type (so the categorical string survives to the
        encoder). Extra request fields are ignored.
        """
        missing = [name for name in keep_columns if name not in features or features[name] is None]
        if missing:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Missing/null required raw covariate(s) for gold-standard "
                    f"SHAP prediction: {missing}"
                ),
            )

        categorical_columns = self._infer_categorical_covariates(
            keep_columns, encoded_feature_columns
        )
        resolved: Dict[str, Any] = {}
        for name in keep_columns:
            raw = features[name]
            if name in categorical_columns:
                if not isinstance(raw, str):
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            f"Raw covariate '{name}' must be a categorical string "
                            f"for an audit-grade SHAP prediction (got {type(raw).__name__})"
                        ),
                    )
                # Categorical RAW covariate — the FeatureBuilder one-hot-encodes
                # it. A non-empty string is a real category; an empty string is
                # an absent value and must fail closed (no fabricated category).
                if not raw.strip():
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            f"Raw covariate '{name}' is an empty string; refusing "
                            "to fabricate a category for an audit-grade SHAP "
                            "prediction"
                        ),
                    )
                resolved[name] = raw
            elif isinstance(raw, (int, float, bool)):
                # Numeric covariate — keep native type (FeatureBuilder medians /
                # one-hot handle int vs float; bool is a valid 0/1 covariate).
                resolved[name] = raw
            else:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"Raw covariate '{name}' must be numeric for an audit-grade "
                        f"SHAP prediction (got "
                        f"{type(raw).__name__})"
                    ),
                )
        return resolved

    async def get_prediction(
        self,
        features: Dict[str, Any],
        model_type: ModelType,
        model_version_id: Optional[str] = None,
        brand: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get a REAL prediction from the BentoML endpoint.

        The feature dict is vectorized into the served model's authoritative
        POSITIONAL order (resolved from /model_info ``feature_columns``) and sent
        in the flat contract ``{"input_data": {"features": [[...]],
        "model_type": ...}}``. This prediction feeds the SHAP explanation AND the
        audit record, so it FAILS CLOSED (503) when the model service is
        unavailable, exposes no feature order, or a required feature is missing —
        it does NOT fabricate a plausible prediction (which would silently
        corrupt audit-grade SHAP output). The previous hardcoded ``0.78``
        demonstration fallback is removed for this reason.

        Multi-model routing (#39): for a gold-standard cohort the per-brand
        serving ``model_name`` (``f"{cohort}_{brand}_goldstd_lr_v1"``) is resolved
        from ``brand`` and sent to BentoML so the shared multi-model service
        routes to the correct per-brand bundle.
        """
        await self._ensure_initialized()

        if not self.bentoml_client:
            raise HTTPException(
                status_code=503,
                detail="Model serving unavailable: BentoML client not configured",
            )

        is_goldstd = model_type in GOLDSTD_COHORT_MODEL_TYPES
        # The serving name routes the shared multi-model service. Gold-standard
        # → per-brand name; legacy → the bare model_type value (single-model).
        serving_name = goldstd_serving_name(model_type, brand) if is_goldstd else model_type.value

        # Resolve the canonical, strictly-validated model feature dict against the
        # ROUTED model's /model_info. SINGLE source of truth for the prediction,
        # SHAP inputs, and audit record — no fabricated value or extra field leaks
        # into audit-grade output. Legacy: {name: float} in /model_info order.
        # Gold-standard (#39): the RAW covariate dict (categorical string kept).
        canonical_features = await self.resolve_canonical_model_features(
            features, model_type, serving_name=serving_name
        )

        try:
            if is_goldstd:
                # Send RAW covariates; the served FeatureBuilder encodes them and
                # the response carries the ENCODED vector SHAP must run over.
                result = await self.bentoml_client.predict(
                    model_name=serving_name,
                    input_data={
                        "model_name": serving_name,
                        "raw_features": [dict(canonical_features)],
                        "model_type": "classification",
                    },
                )
            else:
                ordered_row = [
                    canonical_features[name]
                    for name in canonical_features  # already in model order
                ]
                result = await self.bentoml_client.predict(
                    model_name=serving_name,
                    input_data={"features": [ordered_row], "model_type": "classification"},
                )
        except Exception as e:
            logger.error("BentoML prediction failed for SHAP (%s): %s", serving_name, e)
            raise HTTPException(
                status_code=503,
                detail=f"Model prediction failed for '{model_type.value}'",
            )

        # Multi-model fail-closed (#39): the shared service returns an ``error``
        # field for an unknown/unrouted model_name. Surface it as a clear 502
        # rather than letting it fall through to the generic "no probabilities".
        if result.get("error"):
            logger.error(
                "BentoML returned an error for serving model '%s': %s",
                serving_name,
                result["error"],
            )
            raise HTTPException(
                status_code=502,
                detail=f"Model '{serving_name}' is not served: {result['error']}",
            )

        # Flat contract: ``probabilities`` is a flat positive-class list. A real
        # probability is REQUIRED for the audit-grade SHAP record — we do NOT
        # fabricate a 0.0 from class predictions on a malformed/empty response.
        probs = result.get("probabilities")
        if not isinstance(probs, list) or not probs:
            logger.error(
                "Model '%s' returned no probabilities; refusing to fabricate an "
                "audit-grade SHAP probability.",
                model_type.value,
            )
            raise HTTPException(
                status_code=502,
                detail=(
                    f"Model '{model_type.value}' returned no probabilities; cannot "
                    "produce an audit-grade SHAP prediction"
                ),
            )
        prediction_proba = float(probs[0])

        # The feature dict callers MUST use for SHAP + audit storage. For the
        # gold-standard contract (#39) SHAP must run over the ENCODED numeric
        # vector (NOT the 3 raw covariates), so we use the encoded features the
        # served FeatureBuilder produced and the model actually scored. We FAIL
        # CLOSED if the service did not return them (refusing to silently run
        # SHAP over the wrong/raw vector and label it audit-grade).
        if is_goldstd:
            shap_features = self._encoded_features_from_result(result, model_type)
        else:
            shap_features = canonical_features

        return {
            "prediction_class": "high_propensity" if prediction_proba > 0.5 else "low_propensity",
            "prediction_probability": prediction_proba,
            "model_version_id": (
                result.get("model_id")
                or result.get("_metadata", {}).get("model_name")
                or model_version_id
                or "unknown"
            ),
            # The canonical feature dict for SHAP/audit. Legacy: the validated
            # numeric request features. Gold-standard: the ENCODED vector the
            # model scored (the audit-grade SHAP inputs).
            "model_features": shap_features,
            # Gold-standard SHAP (Option B, #39) runs on the BentoML side, which
            # re-encodes the RAW covariates — so it needs the per-brand serving
            # name and the SAME validated RAW covariates this prediction used.
            # Surfaced here (single source of truth) so the endpoint can thread
            # them into compute_shap without re-deriving them. None on the legacy
            # path (SHAP there runs in-process over the encoded model_features).
            "serving_name": serving_name if is_goldstd else None,
            "raw_features": dict(canonical_features) if is_goldstd else None,
        }

    def _encoded_features_from_result(
        self, result: Dict[str, Any], model_type: ModelType
    ) -> Dict[str, float]:
        """Extract the ENCODED {name: float} SHAP vector from a raw-path response.

        The BentoML raw-covariate path returns ``encoded_features`` (the matrix
        the FeatureBuilder produced) and ``encoded_feature_columns`` (their
        names). SHAP for a gold-standard cohort runs over THIS encoded numeric
        vector, not the 3 raw covariates. FAILS CLOSED (502) when the service
        omits/malforms it — running SHAP over the wrong vector and persisting it
        as audit-grade would be the #532/#576 harm.
        """
        cols = result.get("encoded_feature_columns")
        rows = result.get("encoded_features")
        if not isinstance(cols, list) or not cols or not isinstance(rows, list) or not rows:
            logger.error(
                "Model '%s' raw path returned no encoded feature vector; refusing "
                "to run SHAP over a non-audit-grade vector.",
                model_type.value,
            )
            raise HTTPException(
                status_code=502,
                detail=(
                    f"Model '{model_type.value}' returned no encoded feature vector "
                    "for an audit-grade SHAP explanation"
                ),
            )
        row0 = rows[0]
        if not isinstance(row0, list) or len(row0) != len(cols):
            raise HTTPException(
                status_code=502,
                detail=(
                    f"Model '{model_type.value}' encoded vector shape mismatch "
                    f"(cols={len(cols)}, values={len(row0) if isinstance(row0, list) else 'n/a'})"
                ),
            )
        return {str(name): float(val) for name, val in zip(cols, row0, strict=True)}

    def _prepare_numeric_features(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Convert features to numeric values for model input."""
        numeric_features = {}
        for key, value in features.items():
            if isinstance(value, (int, float)):
                numeric_features[key] = float(value)
            elif isinstance(value, bool):
                numeric_features[key] = 1.0 if value else 0.0
            elif isinstance(value, str):
                # Simple encoding for categorical strings
                numeric_features[key] = hash(value) % 100 / 100.0
            else:
                numeric_features[key] = 0.0
        return numeric_features

    async def _compute_shap_goldstd(
        self,
        features: Dict[str, Any],
        model_type: ModelType,
        model_version_id: str,
        top_k: int,
        serving_name: Optional[str],
        raw_features: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compute gold-standard cohort SHAP on the BentoML side (#39, Option B).

        Calls the service ``/shap`` endpoint with the per-brand serving name and
        the RAW covariates (the service re-encodes them via the bundled
        FeatureBuilder and runs SHAP over the ENCODED vector). Maps the response
        into the SAME shape the legacy path returns (a ``contributions`` list of
        :class:`FeatureContribution`, ``base_value``, ``shap_sum``,
        ``explainer_type``, ``computation_time_ms``) so the FE renders identically.

        FAILS CLOSED (no fabricated SHAP for an audit-grade explanation):

        * 500 if the internal contract is broken (no serving name, no raw
          covariates threaded through — a programming error, not a user error).
        * 503 if the BentoML client is unavailable or the call fails.
        * 502 if the service returns a fail-closed ``error`` or an empty/malformed
          SHAP map.

        ``features`` here is the ENCODED vector ``get_prediction`` produced — used
        only to surface the encoded ``feature_value`` for each contribution; the
        SHAP math runs service-side over the bundle's own re-encoding.
        """
        # The serving name + RAW covariates are the SINGLE source of truth from
        # get_prediction. Missing them is an internal wiring error, not a bad
        # request — fail closed loudly rather than guessing.
        if not serving_name or not isinstance(raw_features, dict) or not raw_features:
            logger.error(
                "Gold-standard SHAP missing serving_name/raw_features for model "
                "%s; refusing to compute SHAP over an unknown vector.",
                model_type.value,
            )
            raise HTTPException(
                status_code=500,
                detail="Internal error: gold-standard SHAP context was not provided",
            )

        if not self.bentoml_client:
            raise HTTPException(
                status_code=503,
                detail="Model serving unavailable: BentoML client not configured",
            )

        try:
            result = await self.bentoml_client.get_shap(
                model_name=serving_name,
                raw_features=[dict(raw_features)],
            )
        except Exception as e:
            logger.error("BentoML /shap call failed for '%s': %s", serving_name, e)
            raise HTTPException(
                status_code=503,
                detail=f"SHAP computation failed for '{model_type.value}'",
            )

        # Multi-model fail-closed: the service sets ``error`` for an unknown
        # model_name / missing FeatureBuilder / explainer failure.
        if result.get("error"):
            logger.error(
                "BentoML /shap returned an error for '%s': %s", serving_name, result["error"]
            )
            raise HTTPException(
                status_code=502,
                detail=f"SHAP unavailable for '{serving_name}': {result['error']}",
            )

        shap_values_map = result.get("shap_values")
        if not isinstance(shap_values_map, dict) or not shap_values_map:
            logger.error(
                "BentoML /shap returned no shap_values for '%s'; refusing to "
                "fabricate an audit-grade SHAP explanation.",
                serving_name,
            )
            raise HTTPException(
                status_code=502,
                detail=(
                    f"Model '{model_type.value}' returned no SHAP values for an "
                    "audit-grade explanation"
                ),
            )

        shap_values_map = {str(k): float(v) for k, v in shap_values_map.items()}
        base_value = float(result.get("base_value", 0.0))
        explainer_type_str = str(result.get("explainer_type") or "LinearExplainer")
        computation_time_ms = float(result.get("computation_time_ms", 0.0))

        # ``features`` is the ENCODED vector from get_prediction — surface its
        # value per contribution so the FE shows the encoded feature value.
        contributions: List[FeatureContribution] = []
        sorted_shap = sorted(shap_values_map.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k]
        for rank, (feature_name, shap_value) in enumerate(sorted_shap, 1):
            contributions.append(
                FeatureContribution(
                    feature_name=feature_name,
                    feature_value=features.get(feature_name),
                    shap_value=shap_value,
                    contribution_direction="positive" if shap_value > 0 else "negative",
                    contribution_rank=rank,
                )
            )

        return {
            "base_value": base_value,
            "contributions": contributions,
            "shap_sum": sum(shap_values_map.values()),
            "explainer_type": explainer_type_str,
            "computation_time_ms": computation_time_ms,
        }

    async def compute_shap(
        self,
        features: Dict[str, Any],
        model_type: ModelType,
        model_version_id: str,
        top_k: int = 5,
        *,
        serving_name: Optional[str] = None,
        raw_features: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Compute SHAP values for a single instance using real SHAP explainer.

        Uses TreeExplainer for tree-based models (fast),
        KernelExplainer for others (slower).

        Gold-standard cohorts (#39, Option B): SHAP is computed on the BentoML
        side via ``/shap`` because the cohort bundle (model + FeatureBuilder)
        lives only in the BentoML container's mount — the API process cannot read
        it. ``serving_name`` (the per-brand registry name) and ``raw_features``
        (the RAW covariates the prediction used) are threaded from the endpoint /
        ``get_prediction`` so the service can re-encode and explain the SAME
        instance. The legacy (MLflow / in-process explainer) path is unchanged for
        the non-gold-standard model types.
        """
        from src.api.dependencies.compute import (
            await_celery_result,
            heavy_offload_enabled,
        )

        await self._ensure_initialized()

        # Gold-standard branch (#39, Option B): delegate to the BentoML /shap
        # endpoint. The legacy path below is untouched for non-goldstd models.
        if model_type in GOLDSTD_COHORT_MODEL_TYPES:
            return await self._compute_shap_goldstd(
                features=features,
                model_type=model_type,
                model_version_id=model_version_id,
                top_k=top_k,
                serving_name=serving_name,
                raw_features=raw_features,
            )

        # Prepare numeric features
        numeric_features = self._prepare_numeric_features(features)

        try:
            # Normalize the SHAP compute to four locals so the inline path and the
            # P2 offload path produce an IDENTICAL response shape downstream.
            if heavy_offload_enabled():
                # P2 offload path (DARK by default): run the heavy SHAP compute on
                # worker_heavy. Feature fetch / prediction / audit stay on the API
                # (light + request-scoped); only the explainer call moves. The task
                # runs the SAME explainer via src.mlops.shap_runner and returns a
                # JSON dict, so the contributions built below are identical.
                # Enqueue by registered task NAME via the existing send_task
                # idiom (src/workers/celery_app.py) so importing the heavy task
                # package — which pulls sklearn/ML libs into the API process via
                # src/tasks/__init__ — is avoided on the offload path.
                from src.workers.celery_app import celery_app

                payload = {
                    "features": numeric_features,
                    "model_type": model_type.value,
                    "model_version_id": model_version_id,
                    "top_k": top_k,
                }
                async_result = celery_app.send_task(
                    "src.tasks.compute_shap_values", args=[payload], queue="shap"
                )
                try:
                    shap_dict = await await_celery_result(
                        async_result, timeout=_SHAP_OFFLOAD_TIMEOUT_SECONDS
                    )
                except TimeoutError:
                    raise HTTPException(
                        status_code=408,
                        detail="SHAP computation timed out; retry shortly.",
                    )
                shap_values_map: Dict[str, float] = shap_dict["shap_values"]
                base_value = shap_dict["base_value"]
                explainer_type_str = shap_dict["explainer_type"]
                computation_time_ms = shap_dict["computation_time_ms"]
            else:
                # P1 inline path (default + fallback): use the real SHAP explainer
                # in-process (it offloads the CPU-bound math to its own thread pool).
                shap_result: SHAPResult = await self.shap_explainer.compute_shap_values(
                    features=numeric_features,
                    model_type=model_type.value,
                    model_version_id=model_version_id,
                    top_k=top_k,
                )
                shap_values_map = shap_result.shap_values
                base_value = shap_result.base_value
                explainer_type_str = shap_result.explainer_type.value
                computation_time_ms = shap_result.computation_time_ms

            # Convert to API response format (identical for both paths)
            contributions = []
            sorted_shap = sorted(shap_values_map.items(), key=lambda x: abs(x[1]), reverse=True)[
                :top_k
            ]

            for rank, (feature_name, shap_value) in enumerate(sorted_shap, 1):
                # Map back to original feature value
                original_value = features.get(feature_name, numeric_features.get(feature_name))
                contributions.append(
                    FeatureContribution(
                        feature_name=feature_name,
                        feature_value=original_value,
                        shap_value=shap_value,
                        contribution_direction="positive" if shap_value > 0 else "negative",
                        contribution_rank=rank,
                    )
                )

            return {
                "base_value": base_value,
                "contributions": contributions,
                "shap_sum": sum(shap_values_map.values()),
                "explainer_type": explainer_type_str,
                "computation_time_ms": computation_time_ms,
            }

        except HTTPException:
            # 408 offload timeout must propagate as-is (other failures -> 500 below).
            raise
        except Exception as e:
            logger.error(f"SHAP computation failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"SHAP computation failed: {str(e)}")

    async def generate_narrative(
        self, patient_id: str, prediction: Dict[str, Any], contributions: List[FeatureContribution]
    ) -> str:
        """
        Generate natural language explanation.

        TODO: Integrate with explainer agent for Claude-powered narratives.
        """
        # For now, generate a structured narrative
        top_factors = ", ".join([c.feature_name.replace("_", " ") for c in contributions[:3]])
        direction = "increases" if contributions[0].shap_value > 0 else "decreases"

        return (
            f"This patient shows {prediction['prediction_class'].replace('_', ' ')} "
            f"(confidence: {prediction['prediction_probability']:.0%}). "
            f"Key factors: {top_factors}. "
            f"The primary driver ({contributions[0].feature_name.replace('_', ' ')}) "
            f"{direction} the prediction by {abs(contributions[0].shap_value):.3f}."
        )

    async def store_audit_record(
        self,
        explanation_id: str,
        patient_id: str,
        model_type: str,
        model_version_id: str,
        features: Dict[str, Any],
        shap_values: Dict[str, float],
        prediction: Dict[str, Any],
    ) -> bool:
        """Store explanation in ``ml_shap_analyses`` for regulatory audit.

        Issue #321 HIGH — write rows with ``analysis_type='local_realtime'``
        and the canonical local-explanation columns (``patient_id``,
        ``entity_type='patient'``, ``entity_id=patient_id``, signed
        ``local_shap_values``, ``prediction_*``, ``model_type``,
        ``model_version_id``, ``explanation_id``, ``request_timestamp``)
        so that ``/explain/history/{patient_id}`` can actually retrieve
        the writes produced by ``/explain/predict``.

        Previously the route called ``ShapAnalysisRepository.store_analysis``,
        which writes ``analysis_type='global'`` plus mean-absolute
        ``global_importance`` only — meaning new realtime explanations
        could not be looked up by patient_id (the new HIGH symptom of #321).
        """
        await self._ensure_initialized()

        if self.shap_repo is None or self.shap_repo.client is None:
            logger.warning("SHAP repository not available, skipping audit storage")
            return False

        try:
            now = datetime.now(timezone.utc).isoformat()
            # Only the columns actually present on ``ml_shap_analyses`` per
            # ``database/ml/mlops_tables.sql`` + migration
            # ``database/ml/011_realtime_shap_audit.sql``. PostgREST rejects
            # inserts that name unknown columns, so we MUST NOT write fields
            # like ``model_type`` / ``model_version_id`` / ``shap_values`` that
            # have no schema column. (The retrieval normalizer still surfaces
            # these via best-effort row.get() so future schema additions can
            # widen the column set without code changes.)
            db_record: Dict[str, Any] = {
                "id": str(uuid.uuid4()),
                "model_registry_id": None,
                "analysis_type": "local_realtime",
                # migration 011 columns
                "explanation_id": explanation_id,
                "patient_id": patient_id,
                "request_timestamp": now,
                "prediction_class": prediction.get("prediction_class"),
                "prediction_probability": prediction.get("prediction_probability"),
                # mlops_tables.sql local-explanation columns
                "entity_type": "patient",
                "entity_id": patient_id,
                # Signed per-feature contributions — canonical column for
                # local explanations (mlops_tables.sql:363).
                "local_shap_values": dict(shap_values),
                "base_value": prediction.get("base_value"),
                "computed_at": now,
                "key_drivers": [
                    name
                    for name, _ in sorted(
                        shap_values.items(), key=lambda x: abs(x[1]), reverse=True
                    )[:5]
                ],
                "computation_method": "TreeExplainer",
            }
            # Strip None to let DB defaults apply.
            db_record = {k: v for k, v in db_record.items() if v is not None}

            result_or_coro = (
                self.shap_repo.client.table(self.shap_repo.table_name).insert(db_record).execute()
            )
            if inspect.isawaitable(result_or_coro):
                await result_or_coro

            logger.info(f"Stored audit record for explanation {explanation_id}")
            return True

        except Exception as e:
            logger.error(f"Error storing audit record: {e}", exc_info=True)
            return False


# Singleton service instance
_shap_service: Optional[RealTimeSHAPService] = None


async def get_shap_service() -> RealTimeSHAPService:
    """Dependency injection for SHAP service."""
    global _shap_service
    if _shap_service is None:
        _shap_service = RealTimeSHAPService()
    return _shap_service


# =============================================================================
# ENDPOINTS
# =============================================================================


@router.post(
    "/predict",
    response_model=ExplainResponse,
    summary="Get prediction with real-time SHAP explanation",
    operation_id="explain_prediction",
    description="""
    Returns a model prediction along with SHAP-based feature explanations.

    **Use Cases:**
    - Field rep needs to explain why a patient was flagged
    - HCP wants to understand recommendation reasoning
    - Regulatory audit requires decision documentation

    **Performance:**
    - TreeExplainer: ~50-150ms (tree-based models)
    - KernelExplainer: ~500-2000ms (other models)

    **Compliance:**
    - Set `store_for_audit=True` to persist explanation for regulatory review
    """,
)
async def explain_prediction(
    request: ExplainRequest,
    background_tasks: BackgroundTasks,
    user: Dict[str, Any] = Depends(require_auth),
) -> ExplainResponse:
    """
    Real-time prediction with SHAP explanation.
    """
    import time
    from contextlib import nullcontext

    from src.api.dependencies.compute import heavy_compute_slot, heavy_offload_enabled

    start_time = time.time()

    # Get service instance
    service = await get_shap_service()

    explanation_id = f"EXPL-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{uuid.uuid4().hex[:8]}"

    # SHAP compute is the heavy, ~1.3 GiB part of this request. On the P1 inline
    # path (DARK default) we hold ONE per-worker heavy-compute slot for the whole
    # request (OOM guard). The slot is acquired on entering the context manager,
    # BEFORE the try below, so a saturated slot raises HeavyComputeSaturated
    # (mapped to 503 + Retry-After by the app exception handler) instead of being
    # swallowed into a 500. SHAP already runs in its own thread pool, so we hold
    # the slot here rather than offloading via run_in_bounded_executor.
    # reuse_if_held=True lets the batch endpoint hold a single slot for its whole
    # fan-out without each inner call contending for (and self-rejecting on) a
    # second slot.
    #
    # On the P2 offload path (flag on) the heavy SHAP runs on worker_heavy, so we
    # must NOT hold the API's reject-fast slot for the duration of the poll —
    # doing so would needlessly 503 concurrent requests while the API is just
    # awaiting a remote result. Use a nullcontext in that case.
    _slot = nullcontext() if heavy_offload_enabled() else heavy_compute_slot(reuse_if_held=True)
    async with _slot:
        try:
            # 1. Get features (from request or Feast). HCP-grain cohorts key on
            # hcp_id (passed as entity_id); patient cohorts on patient_id.
            features = request.features
            if features is None:
                features = await service.get_features(
                    request.patient_id,
                    request.model_type,
                    entity_id=request.hcp_id,
                )

            # 2. Get prediction from BentoML
            prediction = await service.get_prediction(
                features=features,
                model_type=request.model_type,
                model_version_id=request.model_version_id,
                brand=request.brand,
            )

            # Use the canonical, strictly-validated model feature dict for SHAP
            # and audit — NOT the raw request dict, which may carry extra or
            # non-numeric fields that _prepare_numeric_features would fabricate
            # (hash/zero-fill) into audit-grade output. FAIL CLOSED on a broken
            # internal contract rather than silently falling back to raw inputs.
            model_features = prediction.get("model_features")
            if not isinstance(model_features, dict) or not all(
                isinstance(v, (int, float)) and not isinstance(v, bool)
                for v in model_features.values()
            ):
                logger.error(
                    "get_prediction did not return a validated numeric "
                    "model_features dict; refusing SHAP/audit over raw features."
                )
                raise HTTPException(
                    status_code=500,
                    detail="Internal error: model features were not validated for SHAP",
                )

            # 3. Compute SHAP values
            shap_result = await service.compute_shap(
                features=model_features,
                model_type=request.model_type,
                model_version_id=prediction["model_version_id"],
                top_k=request.top_k,
                # Gold-standard SHAP (#39, Option B) runs on the BentoML side and
                # needs the per-brand serving name + the RAW covariates this
                # prediction used. get_prediction surfaces both (None on the
                # legacy path, where these are ignored).
                serving_name=prediction.get("serving_name"),
                raw_features=prediction.get("raw_features"),
            )

            # 4. Generate narrative (if requested)
            narrative = None
            if request.format == ExplanationFormat.NARRATIVE:
                narrative = await service.generate_narrative(
                    patient_id=request.patient_id,
                    prediction=prediction,
                    contributions=shap_result["contributions"],
                )

            # 5. Store audit record (async background task)
            audit_stored = False
            if request.store_for_audit:
                background_tasks.add_task(
                    service.store_audit_record,
                    explanation_id=explanation_id,
                    patient_id=request.patient_id,
                    model_type=request.model_type.value,
                    model_version_id=prediction["model_version_id"],
                    features=model_features,
                    shap_values={
                        c.feature_name: c.shap_value for c in shap_result["contributions"]
                    },
                    prediction=prediction,
                )
                audit_stored = True

            computation_time_ms = (time.time() - start_time) * 1000

            # Mask patient_id in response to protect PII (Phase 3 security enhancement)
            # Original patient_id is preserved in audit records for authorized access
            masked_patient_id = mask_identifier(request.patient_id)

            return ExplainResponse(
                explanation_id=explanation_id,
                request_timestamp=datetime.now(timezone.utc),
                patient_id=masked_patient_id,
                model_type=request.model_type,
                model_version_id=prediction["model_version_id"],
                prediction_class=prediction["prediction_class"],
                prediction_probability=prediction["prediction_probability"],
                base_value=shap_result["base_value"] if request.include_base_value else None,
                top_features=shap_result["contributions"],
                shap_sum=shap_result["shap_sum"],
                narrative_explanation=narrative,
                computation_time_ms=round(computation_time_ms, 2),
                audit_stored=audit_stored,
            )

        except HTTPException:
            # Status-bearing failures (e.g. the 408 SHAP offload timeout bubbling
            # up from service.compute_shap) must propagate as-is, not be
            # re-wrapped into a generic 500 below.
            raise
        except Exception as e:
            logger.error(f"Explanation failed for patient {request.patient_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}") from e


@router.post(
    "/predict/batch",
    response_model=BatchExplainResponse,
    summary="Batch predictions with SHAP explanations",
    operation_id="explain_batch",
    description="Process up to 50 patients in a single request. Useful for pre-computing explanations.",
)
async def explain_batch(
    request: BatchExplainRequest,
    background_tasks: BackgroundTasks,
    user: Dict[str, Any] = Depends(require_auth),
) -> BatchExplainResponse:
    """
    Batch explanation endpoint for multiple patients.
    """
    import time

    from src.api.dependencies.compute import heavy_compute_slot

    start_time = time.time()
    batch_id = f"BATCH-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{uuid.uuid4().hex[:8]}"

    explanations: list[ExplainResponse] = []
    errors: list[dict[str, Any]] = []

    async def process_single(req: ExplainRequest) -> Optional[ExplainResponse]:
        try:
            # The enclosing batch already holds the heavy-compute slot; the inner
            # explain_prediction reuses it (reuse_if_held=True) rather than
            # contending for a second slot.
            return await explain_prediction(req, background_tasks)
        except HTTPException as e:
            # Mask patient_id in error responses to protect PII
            errors.append({"patient_id": mask_identifier(req.patient_id), "error": e.detail})
            return None

    # Hold ONE heavy-compute slot for the whole batch fan-out (OOM guard). An
    # empty batch does no heavy work, so skip slot acquisition (avoids rejecting
    # a trivial empty request under load). If saturated, heavy_compute_slot()
    # raises HeavyComputeSaturated on enter -> 503 + Retry-After.
    if not request.requests:
        total_time_ms = (time.time() - start_time) * 1000
        return BatchExplainResponse(
            batch_id=batch_id,
            total_requests=0,
            successful=0,
            failed=0,
            explanations=[],
            errors=[],
            total_time_ms=round(total_time_ms, 2),
        )

    async with heavy_compute_slot():
        # Sequential on purpose (see docstring): one slot + one SHAP at a time
        # keeps the batch's peak memory at a single explanation's footprint.
        for req in request.requests:
            result = await process_single(req)
            if result:
                explanations.append(result)

    total_time_ms = (time.time() - start_time) * 1000

    return BatchExplainResponse(
        batch_id=batch_id,
        total_requests=len(request.requests),
        successful=len(explanations),
        failed=len(errors),
        explanations=explanations,
        errors=errors,
        total_time_ms=round(total_time_ms, 2),
    )


@router.get(
    "/history/{patient_id}",
    summary="Get explanation history for a patient",
    operation_id="get_explanation_history",
    description="Retrieve past explanations for audit or review purposes.",
)
async def get_explanation_history(
    patient_id: str,
    model_type: Optional[ModelType] = None,
    limit: int = 10,
    user: Dict[str, Any] = Depends(require_auth),
) -> Dict[str, Any]:
    """
    Retrieve historical explanations for a patient.

    Useful for:
    - Audit trail review
    - Understanding prediction evolution over time
    - Debugging model behavior

    Authorization (Finding 1 — BOLA/IDOR): this route returns a patient's
    SHAP explanation (prediction class/prob + signed feature contributions),
    so it requires authentication (``Depends(require_auth)``, mirroring the
    sibling ``/predict`` and ``/predict/batch`` routes) AND enforces
    object-level authorization. The global auth middleware authenticates the
    caller but does NOT authorize them per-object — without the per-row
    ``hcp_id`` scope check below, any authenticated user could read any
    patient's explanation by passing the raw path ``patient_id``.
    Admins / ``'all'``-grant callers see every row; a scoped caller only sees
    rows whose ``hcp_id`` is in their grant set (RLS model in
    ``database/ml/011_realtime_shap_audit.sql``).
    """
    # Mask patient_id in all responses to protect PII
    masked_patient_id = mask_identifier(patient_id)

    try:
        repo = await get_shap_analysis_repository()
        if repo.client is None:
            return {
                "patient_id": masked_patient_id,
                "total_explanations": 0,
                "explanations": [],
                "message": "Database connection not available",
            }

        # Issue #321 HIGH — filter rows by patient_id. The ml_shap_analyses
        # table has a `patient_id` column (per migration
        # database/ml/011_realtime_shap_audit.sql) so the path-param can be
        # used directly; previously the route ignored it. Rows are normalized
        # to the FE ``ExplanationHistoryResponse { explanations: ExplainResponse[] }``
        # contract so the frontend type matches runtime.
        #
        # ``get_shap_analysis_repository`` installs an async Supabase client
        # (see ``src/repositories/shap_analysis.py``), so ``.execute()``
        # returns a coroutine and must be awaited. We detect awaitability so
        # the same code path works with both async clients (live) and the
        # sync ``MagicMock`` chains used by unit tests.
        query = repo.client.table(repo.table_name).select("*").eq("patient_id", patient_id)
        # NOTE: ``model_type`` is not a column on the current
        # ``ml_shap_analyses`` schema (only ``model_registry_id`` FK is), so
        # we cannot push the optional ``model_type`` filter into the
        # supabase query — doing so would 4xx with PostgREST. We retrieve
        # all rows for the patient and apply the filter client-side after
        # row normalization. A schema migration adding ``model_type`` will
        # let us push this back into the query.
        result_or_coro = query.order("request_timestamp", desc=True).limit(limit).execute()
        if inspect.isawaitable(result_or_coro):
            result = await result_or_coro
        else:
            result = result_or_coro

        rows = result.data if result.data else []
        # Object-level authorization (Finding 1 — BOLA/IDOR): drop rows the
        # caller is not entitled to see BEFORE normalization, so an
        # out-of-scope patient's explanation is never returned. The brand/
        # tenant scope used elsewhere has no column on this table; the rows
        # carry ``hcp_id``, which is the scope the schema's RLS uses, so we
        # authorize per-row on ``hcp_id`` (admins / ``'all'`` callers bypass).
        rows = [r for r in rows if _caller_can_view_row(user, r)]
        # Optional client-side ``model_type`` filter (see note above on why
        # this is not pushed into the supabase query).
        if model_type is not None:
            rows = [r for r in rows if r.get("model_type") == model_type.value]
        explanations = [_normalize_history_row(row, masked_patient_id) for row in rows]

        return {
            "patient_id": masked_patient_id,
            "total_explanations": len(explanations),
            "explanations": explanations,
        }

    except HTTPException:
        # Status-bearing failures (e.g. an auth/authorization error) must
        # propagate as-is, not be re-wrapped into the generic 500 below.
        raise
    except Exception as e:
        # Finding 2 (swallow): do NOT return HTTP 200 with the internal error
        # string in the body. Log the detail server-side and raise a proper
        # 500 with a generic message so the client sees a real failure and no
        # internals leak. ``mask_identifier`` keeps the patient_id out of logs.
        logger.error(
            "Error retrieving explanation history for patient=%s: %s",
            masked_patient_id,
            e,
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve explanation history",
        ) from e


async def _keep_columns_for_goldstd_families(
    service: "RealTimeSHAPService",
) -> Dict[str, Optional[List[str]]]:
    """Best-effort ``keep_columns`` (raw covariate names) per gold-standard family.

    SHAP runs over the ENCODED feature vector (one-hot ``geographic_region_west``
    + missingness ``disease_severity__isna`` columns), so the raw feature list a
    user thinks in terms of (``geographic_region``, ``disease_severity``) is
    spread across many encoded columns — the Feature-Importance page then shows
    "geographic region" 4-5× and an ``X``/``X__isna`` twin, reading as
    duplicates. Surfacing the model's ``keep_columns`` lets the FE group the
    encoded columns back to their parent covariate.

    ``keep_columns`` is brand-invariant within a cohort family (every per-brand
    model shares the cohort's covariate schema — only the one-hot VALUES differ),
    so we resolve once per family via the default brand's serving name. This is
    DISPLAY metadata only and fully best-effort: any failure (BentoML down, a
    legacy/bare-estimator model with no keep_columns) yields ``None`` for that
    family and the FE falls back to the flat encoded list. The audit-grade
    ``/predict`` path is untouched.
    """
    result: Dict[str, Optional[List[str]]] = {mt.value: None for mt in GOLDSTD_COHORT_MODEL_TYPES}
    try:
        await service._ensure_initialized()
    except Exception as e:  # noqa: BLE001 — display-only enrichment, never load-bearing
        logger.debug("keep_columns enrichment: service init unavailable: %s", e)
        return result

    client = service.bentoml_client
    if client is None:
        return result

    async def _one(mt: ModelType) -> tuple[str, Optional[List[str]]]:
        try:
            serving_name = goldstd_serving_name(mt, _DEFAULT_GOLDSTD_BRAND)
            info = await client.get_model_info(serving_name)
            keep = info.get("keep_columns") if isinstance(info, dict) else None
            if isinstance(keep, list) and keep:
                return mt.value, [str(c) for c in keep]
        except Exception as e:  # noqa: BLE001 — per-family best-effort
            logger.debug("keep_columns enrichment failed for %s: %s", mt.value, e)
        return mt.value, None

    try:
        pairs = await asyncio.gather(*[_one(mt) for mt in GOLDSTD_COHORT_MODEL_TYPES])
        for family, keep in pairs:
            result[family] = keep
    except Exception as e:  # noqa: BLE001
        logger.debug("keep_columns enrichment gather failed: %s", e)
    return result


@router.get(
    "/models",
    summary="List available models for explanation",
    operation_id="list_explainable_models",
    description="Returns models that support real-time SHAP explanations.",
)
async def list_explainable_models() -> Dict[str, Any]:
    """
    List models with SHAP explainer support.
    """
    service = await get_shap_service()

    # Get cache stats from SHAP explainer
    cache_stats = service.shap_explainer.get_cache_stats()

    # Issue #321 LOW — emit `latest_version` per model so the response matches
    # the FE type ``ExplainableModelInfo`` (frontend/src/types/explain.ts).
    # Enrichment is best-effort: if the registry is unreachable, the field
    # is still present but ``None``.
    latest_versions = await _get_latest_versions_by_model_type()
    # Raw covariate names per gold-standard family (best-effort) so the FE can
    # group the ENCODED SHAP columns back to their parent covariate (kills the
    # one-hot / __isna duplication on the Feature-Importance page). None for
    # legacy types and on any enrichment failure.
    keep_columns_by_family = await _keep_columns_for_goldstd_families(service)

    def _explainer_label(mt: ModelType) -> str:
        # Gold-standard cohorts are CalibratedClassifierCV(LogisticRegression) —
        # SHAP picks LinearExplainer for them (the 'logistic'/'linear' hints in
        # shap_explainer_realtime._get_explainer_type).
        if mt in GOLDSTD_COHORT_MODEL_TYPES:
            return "LinearExplainer"
        if mt in (
            ModelType.PROPENSITY,
            ModelType.RISK_STRATIFICATION,
            ModelType.CHURN_PREDICTION,
        ):
            return "TreeExplainer"
        return "KernelExplainer"

    return {
        "supported_models": [
            {
                "model_type": mt.value,
                "latest_version": latest_versions.get(mt.value),
                "explainer_type": _explainer_label(mt),
                "is_gold_standard": mt in GOLDSTD_COHORT_MODEL_TYPES,
                # Raw covariate names (gold-standard families only; None otherwise)
                # — the FE groups encoded SHAP columns under these parents.
                "keep_columns": keep_columns_by_family.get(mt.value),
                "description": f"SHAP explanations for {mt.value.replace('_', ' ')} predictions",
            }
            for mt in ModelType
        ],
        "total_models": len(ModelType),
        "cache_stats": cache_stats,
    }


# =============================================================================
# COHORT-LEVEL (GLOBAL) FEATURE IMPORTANCE (#39 — option 2)
# =============================================================================


def _entity_source_for_model(model_type: ModelType) -> tuple[str, str, str]:
    """Resolve ``(grain, source_table, id_column)`` for sampling real entity IDs.

    Patient-grain cohorts draw IDs from ``patient_journeys.patient_id``; the
    HCP-grain ``hcp_adoption`` cohort draws from ``hcp_profiles.hcp_id``.
    """
    if model_type == ModelType.HCP_ADOPTION:
        return ("hcp", "hcp_profiles", "hcp_id")
    return ("patient", "patient_journeys", "patient_id")


def _coerce_float(value: Any) -> Optional[float]:
    """Best-effort numeric coercion; None for non-numeric (e.g. string-encoded)."""
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


async def _sample_entity_ids(model_type: ModelType, limit: int) -> List[str]:
    """Fetch up to ``limit`` distinct real entity IDs from the cohort source.

    Ordered deterministically by the id column (a stable prefix, not a random
    draw) — this feeds the /sample-entities PICKER, where a stable list is a
    feature (the dropdown must not churn between loads). The cohort-importance
    aggregation uses ``_sample_entity_ids_random`` instead. NO provenance
    (``is_synthetic=False``) filter: the gold-standard cohorts ARE synthetic-gold,
    so that predicate would exclude every row. Fails LOUD (503) when the source
    is unreachable — this feeds a real importance artifact, not a fabricated one.
    """
    from src.memory.services.factories import get_async_supabase_client

    _, table, id_col = _entity_source_for_model(model_type)
    client = await get_async_supabase_client()
    if client is None:
        raise HTTPException(status_code=503, detail="Entity source unavailable")
    try:
        result = await client.table(table).select(id_col).order(id_col).limit(limit).execute()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Entity source lookup failed: {e}") from e
    seen: set[str] = set()
    ids: List[str] = []
    for row in result.data or []:
        eid = row.get(id_col)
        if eid and eid not in seen:
            seen.add(eid)
            ids.append(eid)
    return ids


async def _sample_entity_ids_random(model_type: ModelType, limit: int) -> Optional[List[str]]:
    """Uniform random draw of entity IDs via the ``sample_entity_ids`` RPC (mig 109).

    Sequential synthetic ids correlate with generation order (frontier-append
    rows land at the tail of the id range), so a sorted prefix is NOT a
    representative sample — the cohort aggregate must draw uniformly. Returns
    ``None`` when the RPC is unavailable (pre-migration DB) so the caller can
    fall back to the deterministic prefix WITH a logged warning — a degraded
    sampling strategy still yields honest per-entity values, unlike a fabricated
    feature vector, so a loud fallback beats a hard 503 here.
    """
    from src.memory.services.factories import get_async_supabase_client

    _, table, _ = _entity_source_for_model(model_type)
    client = await get_async_supabase_client()
    if client is None:
        raise HTTPException(status_code=503, detail="Entity source unavailable")
    try:
        result = await client.rpc(
            "sample_entity_ids",
            {"p_source": table, "p_limit": limit},
        ).execute()
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Random entity sampling RPC unavailable ({e}); using prefix fallback")
        return None
    seen: set[str] = set()
    ids: List[str] = []
    for row in result.data or []:
        eid = row.get("entity_id") if isinstance(row, dict) else row
        if eid and eid not in seen:
            seen.add(eid)
            ids.append(eid)
    return ids or None


def _ranking_stable(
    abs_shap_by_feature: Dict[str, List[float]],
    n: int,
    top_k: int = 5,
    z: float = 1.96,
    negligible_frac: float = 0.02,
) -> bool:
    """True when the top-``top_k`` mean-|SHAP| ranking is separated beyond noise.

    Per-feature lists hold |SHAP| for the entities where the feature surfaced;
    entities where it did not surface contributed ~0, so lists are zero-padded
    to ``n`` (mirrors the cohort-mean denominator). The ranking is "stable"
    when every adjacent pair in the top-k is separated by more than
    ``z * SE(gap)`` — i.e. another draw is unlikely to reorder it. Two kinds
    of pair are exempt, because their order carries no insight and a true tie
    can never be statistically separated (it would force every sample to the
    cap):

    - jointly negligible: BOTH means < ``negligible_frac`` of the top mean;
    - confirmed practical tie: the upper confidence bound of the gap
      (``gap + z * SE(gap)``) is below that same threshold — the importances
      are KNOWN to differ by less than matters, even though each feature is
      individually non-negligible. A noisy observed tie (large SE) is NOT
      exempt: there the true gap may be material and more entities genuinely
      help.
    """
    if n < 2 or not abs_shap_by_feature:
        return False
    means: Dict[str, float] = {f: sum(v) / n for f, v in abs_shap_by_feature.items()}
    ordered = sorted(means, key=lambda f: means[f], reverse=True)[:top_k]
    top_mean = means[ordered[0]]
    if top_mean <= 0.0:
        return False
    ses: Dict[str, float] = {}
    for f in ordered:
        vals = abs_shap_by_feature[f] + [0.0] * (n - len(abs_shap_by_feature[f]))
        var = sum((x - means[f]) ** 2 for x in vals) / (n - 1)
        ses[f] = (var / n) ** 0.5
    floor = negligible_frac * top_mean
    for hi, lo in zip(ordered, ordered[1:], strict=False):
        if means[hi] < floor and means[lo] < floor:
            continue
        gap = means[hi] - means[lo]
        se_gap = (ses[hi] ** 2 + ses[lo] ** 2) ** 0.5
        if gap + z * se_gap < floor:
            continue
        if gap <= z * se_gap:
            return False
    return True


async def _resolve_model_registry_id(model_name: str) -> Optional[str]:
    """Resolve a serving ``model_name`` to its latest ``ml_model_registry.id``."""
    from src.memory.services.factories import get_async_supabase_client

    client = await get_async_supabase_client()
    if client is None:
        return None
    try:
        result = await (
            client.table("ml_model_registry")
            .select("id,registered_at")
            .eq("model_name", model_name)
            .order("registered_at", desc=True)
            .limit(1)
            .execute()
        )
    except Exception as e:  # noqa: BLE001
        logger.debug(f"registry id lookup failed for {model_name}: {e}")
        return None
    rows = result.data or []
    return rows[0]["id"] if rows else None


async def _load_global_importance_row(model_registry_id: str) -> Optional[Dict[str, Any]]:
    """Read the latest stored ``analysis_type='global'`` row for a model."""
    from src.memory.services.factories import get_async_supabase_client

    client = await get_async_supabase_client()
    if client is None:
        return None
    try:
        result = await (
            client.table("ml_shap_analyses")
            .select("*")
            .eq("model_registry_id", model_registry_id)
            .eq("analysis_type", "global")
            .order("computed_at", desc=True)
            .limit(1)
            .execute()
        )
    except Exception as e:  # noqa: BLE001
        logger.debug(f"global importance read failed for {model_registry_id}: {e}")
        return None
    rows = result.data or []
    return rows[0] if rows else None


async def _store_global_importance_row(
    model_registry_id: Optional[str],
    model_type: ModelType,
    agg: Dict[str, Any],
    max_stored_points: int = 50,
) -> None:
    """Persist a computed global-importance aggregate as an ``analysis_type='global'``
    row in ``ml_shap_analyses`` (durable cache; no new table — the JSONB
    ``global_importance`` column carries the rich per-feature payload).
    """
    from src.memory.services.factories import get_async_supabase_client

    client = await get_async_supabase_client()
    if client is None or model_registry_id is None:
        return
    points: Dict[str, List[tuple]] = agg["points"]
    gi: Dict[str, Any] = {}
    for feat in agg["features"]:
        name = feat["feature_name"]
        gi[name] = {
            "mean_abs_shap": feat["mean_abs_shap"],
            "mean_shap": feat["mean_shap"],
            "mean_feature_value": feat["mean_feature_value"],
            "contribution_rank": feat["contribution_rank"],
            "points": [{"s": s, "v": v} for (s, v) in points.get(name, [])[:max_stored_points]],
        }
    # Sampling provenance rides in the same JSONB under a reserved dunder key
    # (no real feature can be named "__sampling__"); the parser skips "__" keys.
    gi["__sampling__"] = {
        "sampling_method": agg.get("sampling_method"),
        "stability_achieved": agg.get("stability_achieved"),
        "stopping_reason": agg.get("stopping_reason"),
    }
    record: Dict[str, Any] = {
        "id": str(uuid.uuid4()),
        "model_registry_id": model_registry_id,
        "analysis_type": "global",
        "global_importance": gi,
        "base_value": agg["base_value"],
        "sample_size": agg["sample_size"],
        "computation_method": agg["computation_method"],
        "key_drivers": [f["feature_name"] for f in agg["features"][:5]],
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "entity_type": "hcp" if model_type == ModelType.HCP_ADOPTION else "patient",
        # ``data_split`` is the ``data_split_type`` ENUM
        # {train,validation,test,holdout,unassigned} (database/core/
        # e2i_ml_complete_v3_schema.sql). A cohort-aggregate is not a designated
        # split, so the honest value is ``unassigned``. The previous
        # ``"synthetic"`` is NOT an enum member -> every INSERT was rejected by
        # Postgres and swallowed by the except below, so this durable cache never
        # persisted and /explain/global recomputed the full SHAP aggregate on
        # every load. (Provenance is already carried by is_synthetic on the
        # source rows / the model registry, not by this split label.)
        "data_split": "unassigned",
    }
    record = {k: v for k, v in record.items() if v is not None}
    try:
        result_or_coro = client.table("ml_shap_analyses").insert(record).execute()
        if inspect.isawaitable(result_or_coro):
            await result_or_coro
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Could not persist global importance row: {e}")


def _row_to_global_agg(row: Dict[str, Any]) -> Dict[str, Any]:
    """Parse a stored global row's JSONB back into the in-memory aggregate shape."""
    gi = row.get("global_importance") or {}
    features: List[Dict[str, Any]] = []
    points: Dict[str, List[tuple]] = {}
    sampling_meta: Dict[str, Any] = {}
    if isinstance(gi, dict):
        for name, detail in gi.items():
            if name.startswith("__"):
                # Reserved metadata keys (e.g. "__sampling__"), not features.
                if name == "__sampling__" and isinstance(detail, dict):
                    sampling_meta = detail
                continue
            if isinstance(detail, dict):
                features.append(
                    {
                        "feature_name": name,
                        "mean_abs_shap": float(detail.get("mean_abs_shap", 0.0)),
                        "mean_shap": float(detail.get("mean_shap", 0.0)),
                        "mean_feature_value": (
                            float(detail["mean_feature_value"])
                            if detail.get("mean_feature_value") is not None
                            else None
                        ),
                        "contribution_rank": int(detail.get("contribution_rank", 0)),
                    }
                )
                points[name] = [
                    (float(p["s"]), (float(p["v"]) if p.get("v") is not None else None))
                    for p in detail.get("points", [])
                    if isinstance(p, dict) and p.get("s") is not None
                ]
            else:
                # Legacy bare-number shape (feature -> mean_abs): keep it usable.
                features.append(
                    {
                        "feature_name": name,
                        "mean_abs_shap": abs(float(detail)) if detail is not None else 0.0,
                        "mean_shap": float(detail) if detail is not None else 0.0,
                        "mean_feature_value": None,
                        "contribution_rank": 0,
                    }
                )
                points[name] = []
    features.sort(key=lambda f: f["mean_abs_shap"], reverse=True)
    for idx, feat in enumerate(features):
        if not feat.get("contribution_rank"):
            feat["contribution_rank"] = idx + 1
    return {
        "features": features,
        "points": points,
        "base_value": (float(row["base_value"]) if row.get("base_value") is not None else None),
        "sample_size": int(row.get("sample_size") or 0),
        "computation_method": row.get("computation_method") or "LinearExplainer",
        "computed_at": row.get("computed_at"),
        # None for legacy rows written before mig-109 sampling provenance.
        "sampling_method": sampling_meta.get("sampling_method"),
        "stability_achieved": sampling_meta.get("stability_achieved"),
        "stopping_reason": sampling_meta.get("stopping_reason"),
    }


async def _compute_global_importance(
    model_type: ModelType,
    brand: str,
    sample_size: int,
    background_tasks: BackgroundTasks,
    max_sample_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute cohort-level mean |SHAP| by explaining a random sample of entities.

    Sampling: a uniform random draw (``sample_entity_ids`` RPC, mig 109; loud
    prefix fallback pre-migration). Sizing: ``sample_size`` is the MINIMUM;
    after reaching it the loop keeps sampling until the top-feature ranking is
    stable beyond sampling noise (``_ranking_stable``) or ``max_sample_size``
    is hit — sized for stability, not a fixed constant.

    SHAP is the heavy (~1.3 GiB) part of each call, so the sample is processed
    SEQUENTIALLY under ONE held heavy-compute slot — concurrent fan-out would OOM
    the box (this mirrors ``/predict/batch``, which is sequential on purpose).
    Entities that fail their Feast lookup (a real ~1/8 miss rate) are SKIPPED, not
    fabricated, and ``sample_size`` reports the honest ``n_succeeded``.
    """
    from collections import defaultdict

    from src.api.dependencies.compute import heavy_compute_slot

    max_n = max(max_sample_size or sample_size, sample_size)
    grain, _, _ = _entity_source_for_model(model_type)
    # Over-sample to absorb Feast misses so we can still reach the target count.
    pool_size = max(max_n * 2, max_n + 8)
    sampling_method = "random"
    candidates = await _sample_entity_ids_random(model_type, pool_size)
    if candidates is None:
        sampling_method = "prefix_fallback"
        candidates = await _sample_entity_ids(model_type, pool_size)
    if not candidates:
        raise HTTPException(
            status_code=503,
            detail=f"No sample entities available for {model_type.value}",
        )

    abs_sum: Dict[str, float] = defaultdict(float)
    signed_sum: Dict[str, float] = defaultdict(float)
    val_sum: Dict[str, float] = defaultdict(float)
    val_n: Dict[str, int] = defaultdict(int)
    points: Dict[str, List[tuple]] = defaultdict(list)
    abs_values: Dict[str, List[float]] = defaultdict(list)
    base_values: List[float] = []
    n_succeeded = 0
    stability_achieved = False
    stopping_reason = "candidates_exhausted"

    async with heavy_compute_slot():
        for entity_id in candidates:
            if n_succeeded >= max_n:
                stopping_reason = "max_sample_size_reached"
                break
            req = ExplainRequest(
                patient_id=entity_id,
                hcp_id=entity_id if grain == "hcp" else None,
                model_type=model_type,
                brand=brand,
                # Explicit None for the optionals — runtime defaults, but named
                # so mypy (no pydantic plugin) sees every field supplied.
                model_version_id=None,
                features=None,
                format=ExplanationFormat.TOP_K,
                # 20 is the request cap (ExplainRequest.top_k le=20) and >= the
                # encoded feature counts of every current goldstd model (9 patient
                # / 19 hcp) -> the FULL vector per entity, no truncation. Were a
                # future model to exceed this, the n_succeeded denominator above
                # keeps the cohort mean honest (truncated -> treated as ~0), never
                # inflated.
                top_k=20,
                store_for_audit=False,  # don't flood ml_shap_analyses with sampling rows
            )
            try:
                # Inner call reuses the held slot (reuse_if_held=True) — one SHAP
                # at a time, single-explanation peak memory.
                resp = await explain_prediction(req, background_tasks)
            except HTTPException:
                continue  # skip Feast misses / per-entity failures (honest n_succeeded)
            n_succeeded += 1
            if resp.base_value is not None:
                base_values.append(resp.base_value)
            for c in resp.top_features:
                abs_sum[c.feature_name] += abs(c.shap_value)
                signed_sum[c.feature_name] += c.shap_value
                abs_values[c.feature_name].append(abs(c.shap_value))
                fv = _coerce_float(c.feature_value)
                if fv is not None:
                    val_sum[c.feature_name] += fv
                    val_n[c.feature_name] += 1
                points[c.feature_name].append((c.shap_value, fv))
            # Adaptive stop: past the minimum, quit as soon as the top-feature
            # ranking is separated beyond sampling noise — sized for stability.
            if n_succeeded >= sample_size and _ranking_stable(abs_values, n_succeeded):
                stability_achieved = True
                stopping_reason = "stable"
                break

    if n_succeeded == 0:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Could not explain any sampled entity for "
                f"{model_type.value}/{brand} (feature store / serving unavailable)"
            ),
        )

    # A run that ended at the cap (or ran out of candidates) may still have
    # reached a stable ranking — report that honestly rather than only when the
    # early-stop fired.
    if not stability_achieved and n_succeeded >= 2:
        stability_achieved = _ranking_stable(abs_values, n_succeeded)

    # No silent under-sampling: if Feast misses left us short of the target, the
    # honest n_succeeded still flows to the response — but surface it in logs so a
    # degraded run isn't invisible.
    if n_succeeded < sample_size:
        logger.warning(
            "Global importance for %s/%s reached only %d/%d entities "
            "(remaining candidates exhausted / feature-store misses).",
            model_type.value,
            brand,
            n_succeeded,
            sample_size,
        )

    features: List[Dict[str, Any]] = []
    # Cohort MEAN: divide every feature's SHAP sum by the FULL sample size
    # (n_succeeded), not by the count of entities where the feature surfaced in
    # top_k. A feature absent from an entity's top_k contributed ~0 there, so the
    # cohort mean must treat it as 0 — dividing by a per-feature count would
    # inflate sparsely-surfaced features. (With top_k >= the encoded feature
    # count this is moot today, but it's correct-by-construction.)
    for name, total_abs in abs_sum.items():
        features.append(
            {
                "feature_name": name,
                "mean_abs_shap": total_abs / n_succeeded,
                "mean_shap": signed_sum[name] / n_succeeded,
                # Mean of OBSERVED values only (a truncated feature's value is
                # unknown, not zero) — divide by the count actually seen.
                "mean_feature_value": (val_sum[name] / val_n[name]) if val_n[name] else None,
            }
        )
    features.sort(key=lambda f: f["mean_abs_shap"], reverse=True)
    for idx, feat in enumerate(features):
        feat["contribution_rank"] = idx + 1

    return {
        "features": features,
        "points": dict(points),
        "base_value": (sum(base_values) / len(base_values)) if base_values else None,
        "sample_size": n_succeeded,
        "computation_method": "LinearExplainer",
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "sampling_method": sampling_method,
        "stability_achieved": stability_achieved,
        "stopping_reason": stopping_reason,
    }


@router.get(
    "/sample-entities",
    response_model=SampleEntitiesResponse,
    summary="Real entity IDs for the per-entity SHAP picker",
    operation_id="sample_explain_entities",
    description=(
        "Returns real cohort entity identifiers (patient_id, or hcp_id for the "
        "hcp_adoption cohort) so the UI can offer a picker instead of free-text entry."
    ),
)
async def sample_explain_entities(
    model_type: ModelType,
    limit: int = Query(default=25, ge=1, le=200),
    user: Dict[str, Any] = Depends(require_auth),
) -> SampleEntitiesResponse:
    # NOTE on PII: unlike /predict (which masks patient_id in its response), this
    # endpoint returns RAW entity IDs on purpose. They are the picker's selection
    # keys — the UI sends the chosen id straight back to /predict, so a masked id
    # would be unusable (it would not resolve in Feast) and break the feature.
    # The cohort data here is synthetic-gold (scvpt_*/scvhcp_* are synthetic
    # identifiers, not real PII) and the route is auth-gated. Reasoned exception
    # to the response-masking convention, not an oversight.
    grain, _, id_col = _entity_source_for_model(model_type)
    ids = await _sample_entity_ids(model_type, limit)
    return SampleEntitiesResponse(
        model_type=model_type,
        grain=grain,
        id_field=id_col,
        entities=ids,
    )


@router.get(
    "/global",
    response_model=GlobalFeatureImportanceResponse,
    summary="Cohort-level (global) SHAP feature importance",
    operation_id="global_feature_importance",
    description=(
        "Mean |SHAP| feature importance aggregated over a uniform RANDOM sample "
        "of real cohort entities for one per-brand gold-standard model. "
        "``sample_size`` is the minimum; sampling continues adaptively until the "
        "top-feature ranking is stable beyond sampling noise or "
        "``max_sample_size`` is reached. Reads a durable precomputed row when "
        "available (instant); otherwise computes it once (sequential under the "
        "heavy-compute slot) and stores it. ``refresh=true`` forces recompute. "
        "Only gold-standard cohorts are supported (the legacy demo model types "
        "have no deployed model)."
    ),
)
async def global_feature_importance(
    background_tasks: BackgroundTasks,
    model_type: ModelType,
    brand: str = Query(default=_DEFAULT_GOLDSTD_BRAND),
    sample_size: int = Query(
        default=25,
        ge=5,
        le=60,
        description="Minimum entities to explain; adaptive sizing may sample more",
    ),
    max_sample_size: Optional[int] = Query(
        default=None,
        ge=5,
        le=_MAX_SAMPLE_SIZE_CEILING,
        description=(
            "Hard cap for adaptive sizing (latency/memory budget). Defaults per "
            "model type: 200 for persistence/discontinuation (mid-tail ranking "
            "gaps need larger n to separate), 60 otherwise."
        ),
    ),
    max_points: int = Query(default=30, ge=1, le=60),
    refresh: bool = Query(default=False),
    user: Dict[str, Any] = Depends(require_auth),
) -> GlobalFeatureImportanceResponse:
    if max_sample_size is None:
        max_sample_size = _default_max_sample_size(model_type)
    if max_sample_size < sample_size:
        raise HTTPException(
            status_code=422,
            detail="max_sample_size must be >= sample_size",
        )
    if model_type not in GOLDSTD_COHORT_MODEL_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                "Global feature importance is only available for the gold-standard "
                f"cohorts: {[m.value for m in GOLDSTD_COHORT_MODEL_TYPES]}"
            ),
        )
    # Validates brand (422 on unknown) and resolves the per-brand serving name.
    model_name = goldstd_serving_name(model_type, brand)
    registry_id = await _resolve_model_registry_id(model_name)

    cached = False
    agg: Optional[Dict[str, Any]] = None
    if not refresh and registry_id:
        row = await _load_global_importance_row(registry_id)
        if row is not None:
            parsed = _row_to_global_agg(row)
            # A stored row with no features / zero sample is degenerate (broken or
            # a legacy demo row) — DON'T serve it as a real cohort importance;
            # fall through and recompute instead of returning an empty/zero result.
            if parsed["features"] and parsed["sample_size"] > 0:
                agg = parsed
                cached = True
    if agg is None:
        agg = await _compute_global_importance(
            model_type, brand, sample_size, background_tasks, max_sample_size=max_sample_size
        )
        # The compute path already 503s when n_succeeded == 0; this guards the
        # rarer "explained N entities but every top_features list was empty" case
        # so we never store/serve a featureless aggregate.
        if not agg["features"]:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Computed no feature contributions for {model_type.value}/{brand} "
                    "(serving returned empty explanations)"
                ),
            )
        await _store_global_importance_row(registry_id, model_type, agg)

    features = [
        GlobalImportanceFeature(
            feature_name=f["feature_name"],
            mean_abs_shap=f["mean_abs_shap"],
            mean_shap=f["mean_shap"],
            mean_feature_value=f["mean_feature_value"],
            contribution_rank=f["contribution_rank"],
        )
        for f in agg["features"]
    ]
    # Real beeswarm distribution: per-entity points for the TOP features only.
    top_names = [f["feature_name"] for f in agg["features"][:8]]
    out_points: List[GlobalImportancePoint] = []
    for name in top_names:
        for shap_value, feature_value in agg["points"].get(name, [])[:max_points]:
            out_points.append(
                GlobalImportancePoint(
                    feature_name=name,
                    shap_value=shap_value,
                    feature_value=feature_value,
                )
            )

    computed_at = agg.get("computed_at")
    if isinstance(computed_at, str):
        try:
            computed_at_dt = datetime.fromisoformat(computed_at.replace("Z", "+00:00"))
        except ValueError:
            computed_at_dt = datetime.now(timezone.utc)
    elif isinstance(computed_at, datetime):
        computed_at_dt = computed_at
    else:
        computed_at_dt = datetime.now(timezone.utc)

    return GlobalFeatureImportanceResponse(
        model_type=model_type,
        brand=brand,
        model_name=model_name,
        base_value=agg["base_value"],
        sample_size=int(agg["sample_size"]),
        requested_sample_size=sample_size,
        computation_method=agg["computation_method"],
        computed_at=computed_at_dt,
        cached=cached,
        sampling_method=agg.get("sampling_method"),
        stability_achieved=agg.get("stability_achieved"),
        stopping_reason=agg.get("stopping_reason"),
        features=features,
        points=out_points,
    )


@router.get(
    "/health", summary="Health check for interpretability service", operation_id="health_check"
)
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint for the interpretability service.
    """
    service = await get_shap_service()
    await service._ensure_initialized()

    # Check each dependency
    bentoml_status = "connected" if service.bentoml_client else "not_configured"
    feast_status = "connected" if service.feast_client else "not_configured"
    shap_status = "loaded" if service.shap_explainer else "not_loaded"
    db_status = "connected" if service.shap_repo and service.shap_repo.client else "not_configured"

    # Overall health
    is_healthy = shap_status == "loaded"  # SHAP is the core requirement

    return {
        "status": "healthy" if is_healthy else "degraded",
        "service": "real-time-shap-api",
        "version": "4.2.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dependencies": {
            "bentoml": bentoml_status,
            "feast": feast_status,
            "shap_explainer": shap_status,
            "ml_shap_analyses_db": db_status,
        },
        "cache_stats": service.shap_explainer.get_cache_stats() if service.shap_explainer else {},
    }
