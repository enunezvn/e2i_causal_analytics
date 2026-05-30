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

import inspect
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.api.dependencies.auth import require_auth

# Real implementations
from src.api.dependencies.bentoml_client import BentoMLClient, get_bentoml_client
from src.api.schemas.errors import ErrorResponse, ValidationErrorResponse
from src.api.utils.data_masking import mask_identifier
from src.feature_store.feast_client import FeastClient, get_feast_client
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
    """Supported model types for SHAP explanation."""

    PROPENSITY = "propensity"
    RISK_STRATIFICATION = "risk_stratification"
    NEXT_BEST_ACTION = "next_best_action"
    CHURN_PREDICTION = "churn_prediction"


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
        result = await (
            client.table("ml_model_registry")
            .select("model_name,model_version,registered_at")
            .in_("model_name", list(versions.keys()))
            .order("registered_at", desc=True)
            .execute()
        )

        rows: List[Dict[str, Any]] = result.data or []
        # Order desc by registered_at — first row per model_name wins.
        for r in rows:
            name = r.get("model_name")
            if name in versions and versions[name] is None:
                versions[name] = r.get("model_version")
        return versions

    except Exception as e:
        # Surface as debug — this is best-effort enrichment, not load-bearing.
        logger.debug(f"latest_version enrichment unavailable: {e}")
        return versions


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

    async def get_features(self, patient_id: str, model_type: ModelType) -> Dict[str, Any]:
        """Retrieve features from Feast feature store."""
        await self._ensure_initialized()

        if self.feast_client:
            try:
                # Map model type to feature refs
                feature_refs = self._get_feature_refs_for_model(model_type)

                features_dict = await self.feast_client.get_online_features(
                    entity_rows=[{"patient_id": patient_id}],
                    feature_refs=feature_refs,
                    full_feature_names=False,
                )

                # Convert list values to single values (since we're querying one patient)
                return {k: v[0] if v else None for k, v in features_dict.items()}

            except Exception as e:
                logger.warning(f"Feast feature retrieval failed, using fallback: {e}")

        # Fallback: return default features for demonstration
        return self._get_default_features()

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
        """Default features for fallback/testing."""
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

    async def get_prediction(
        self,
        features: Dict[str, Any],
        model_type: ModelType,
        model_version_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get prediction from BentoML endpoint."""
        await self._ensure_initialized()

        if self.bentoml_client:
            try:
                # Prepare numeric features for model
                numeric_features = self._prepare_numeric_features(features)

                result = await self.bentoml_client.predict(
                    model_name=model_type.value,
                    input_data={"features": [list(numeric_features.values())]},
                )

                # Extract prediction from BentoML response
                prediction_proba = result.get("predictions", [[0.5]])[0]
                if isinstance(prediction_proba, list):
                    prediction_proba = (
                        prediction_proba[1] if len(prediction_proba) > 1 else prediction_proba[0]
                    )

                return {
                    "prediction_class": "high_propensity"
                    if prediction_proba > 0.5
                    else "low_propensity",
                    "prediction_probability": float(prediction_proba),
                    "model_version_id": result.get("_metadata", {}).get(
                        "model_name", model_version_id or "v2.3.1-prod"
                    ),
                }

            except Exception as e:
                logger.warning(f"BentoML prediction failed, using fallback: {e}")

        # Fallback: mock prediction
        return {
            "prediction_class": "high_propensity",
            "prediction_probability": 0.78,
            "model_version_id": model_version_id or "v2.3.1-prod",
        }

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

    async def compute_shap(
        self, features: Dict[str, Any], model_type: ModelType, model_version_id: str, top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Compute SHAP values for a single instance using real SHAP explainer.

        Uses TreeExplainer for tree-based models (fast),
        KernelExplainer for others (slower).
        """
        from src.api.dependencies.compute import (
            await_celery_result,
            heavy_offload_enabled,
        )

        await self._ensure_initialized()

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
            # 1. Get features (from request or Feast)
            features = request.features
            if features is None:
                features = await service.get_features(request.patient_id, request.model_type)

            # 2. Get prediction from BentoML
            prediction = await service.get_prediction(
                features=features,
                model_type=request.model_type,
                model_version_id=request.model_version_id,
            )

            # 3. Compute SHAP values
            shap_result = await service.compute_shap(
                features=features,
                model_type=request.model_type,
                model_version_id=prediction["model_version_id"],
                top_k=request.top_k,
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
                    features=features,
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
) -> Dict[str, Any]:
    """
    Retrieve historical explanations for a patient.

    Useful for:
    - Audit trail review
    - Understanding prediction evolution over time
    - Debugging model behavior
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

    except Exception as e:
        logger.error(f"Error retrieving explanation history: {e}")
        return {
            "patient_id": masked_patient_id,
            "total_explanations": 0,
            "explanations": [],
            "error": str(e),
        }


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

    return {
        "supported_models": [
            {
                "model_type": mt.value,
                "latest_version": latest_versions.get(mt.value),
                "explainer_type": "TreeExplainer"
                if mt
                in [ModelType.PROPENSITY, ModelType.RISK_STRATIFICATION, ModelType.CHURN_PREDICTION]
                else "KernelExplainer",
                "description": f"SHAP explanations for {mt.value.replace('_', ' ')} predictions",
            }
            for mt in ModelType
        ],
        "total_models": len(ModelType),
        "cache_stats": cache_stats,
    }


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
