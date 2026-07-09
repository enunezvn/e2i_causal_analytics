"""Model Prediction API Routes.

This module provides REST endpoints for model inference via BentoML.

Endpoints:
----------
- POST /api/models/predict - Single prediction
- POST /api/models/predict/batch - Batch predictions
- GET /api/models/{model_name}/health - Model health check
- GET /api/models/{model_name}/info - Model metadata
- GET /api/models/status - All models status

Author: E2I Causal Analytics Team
Version: 1.0.0
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from src.api.dependencies.auth import require_auth
from src.api.dependencies.bentoml_client import BentoMLClient, get_bentoml_client
from src.api.dependencies.durable_job_store import DurableJobStore
from src.api.schemas.errors import ErrorResponse, ValidationErrorResponse
from src.feature_store.feast_client import FeastClient, get_feast_client
from src.feature_store.model_feature_refs import MODEL_FEATURE_REFS as _MODEL_FEATURE_REFS
from src.feature_store.model_feature_refs import feature_refs_for_model as _feature_refs_for_model
from src.feature_store.online_feature_presence import missing_or_null_feature_fields

logger = logging.getLogger(__name__)


async def _resolve_production_model_names(limit: int = 50) -> List[str]:
    """Resolve REAL production model names from ``ml_model_registry``.

    Drives the model-status selector from the registry instead of fictional
    hardcoded handles (``churn_model``/``conversion_model``/``causal_model``,
    which are not registered and never resolve). Returns the names of REAL models
    (``is_synthetic=false``) at ``stage IN ('production','staging')`` — the 2 legacy
    ``csu_treatment_initiation_lr_*_v1`` production models PLUS the 12 gold-standard
    ``*_goldstd_lr_v1`` staging models (which were previously invisible). The 360
    synthetic experiment artifacts stay excluded via ``is_synthetic=false``.

    Best-effort: returns ``[]`` if the registry is unreachable so the caller can
    surface an honest "no models" state rather than fabricating handles. Tests
    monkey-patch this function directly.
    """
    try:
        from src.memory.services.factories import get_async_supabase_client

        client = await get_async_supabase_client()
        if client is None:
            return []

        result = await (
            client.table("ml_model_registry")
            .select("model_name")
            # Surface BOTH production and staging real models: the 12 gold-standard
            # models are registered at stage='staging' and were invisible (only the
            # 2 legacy csu_* production models showed). is_synthetic=False still
            # excludes the 360 synthetic experiment artifacts (verified: 14 models).
            .in_("stage", ["production", "staging"])
            .eq("is_synthetic", False)
            .order("registered_at", desc=True)
            .limit(limit)
            .execute()
        )
        rows: List[Dict[str, Any]] = result.data or []
        # De-dup while preserving order (a model may have multiple versions).
        seen: set[str] = set()
        names: List[str] = []
        for r in rows:
            name = r.get("model_name")
            if name and name not in seen:
                seen.add(name)
                names.append(name)
        return names
    except Exception as e:
        logger.warning("Could not resolve production models from registry: %s", e)
        return []


async def _resolve_feature_order(client: "BentoMLClient", model_name: str) -> List[str]:
    """Resolve the served model's authoritative ordered feature names.

    The live BentoML service expects ``features`` as a POSITIONAL numeric matrix
    ordered by the model's own ``feature_columns`` (the preprocessor input
    order, or the estimator's ``feature_names_in_``). The service exposes this
    via ``POST /model_info`` -> ``feature_columns``. We fetch it from the model
    itself rather than guessing/hardcoding an order (the repo has several
    divergent feature lists; only the bundled model knows its real order).

    Fails CLOSED (503) when the model exposes no feature order — never invents a
    positional order, which would silently feed the model a mis-ordered vector
    presented as a real prediction.
    """
    try:
        info = await client.get_model_info(model_name)
    except Exception as e:
        logger.error("Could not fetch model_info for feature order (model=%s): %s", model_name, e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model metadata unavailable for '{model_name}'",
        )

    columns = info.get("feature_columns")
    if not columns or not isinstance(columns, list):
        logger.error(
            "Model '%s' exposes no feature_columns order via /model_info; refusing to "
            "vectorize a feature dict against an unknown positional order.",
            model_name,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"Model '{model_name}' does not expose a feature order; cannot vectorize "
                "feature dictionary"
            ),
        )
    return [str(c) for c in columns]


def _vectorize_feature_dict(
    features: Dict[str, Any], feature_order: List[str], *, context: str
) -> List[float]:
    """Build a single ordered numeric row from a feature dict + canonical order.

    Each value is read by name in ``feature_order``. A missing or null required
    feature FAILS CLOSED with a 422 — no silent zero-fill (which would fabricate
    a plausible-but-wrong prediction). Extra keys not in the order are ignored.
    Non-numeric values raise a 422 with the offending field named.
    """
    missing = [name for name in feature_order if name not in features or features[name] is None]
    if missing:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Missing required feature(s) for {context}: {missing}. "
                f"Expected features (in order): {feature_order}"
            ),
        )
    row: List[float] = []
    for name in feature_order:
        value = features[name]
        try:
            row.append(float(value))
        except (TypeError, ValueError):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Feature '{name}' is not numeric (got {value!r}) for {context}",
            )
    return row


router = APIRouter(
    prefix="/api/models",
    tags=["Model Predictions"],
    responses={
        401: {"model": ErrorResponse, "description": "Authentication required"},
        404: {"model": ErrorResponse, "description": "Model not found"},
        422: {"model": ValidationErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
        503: {"model": ErrorResponse, "description": "Model service unavailable"},
    },
)


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================


class PredictionRequest(BaseModel):
    """Request schema for model prediction."""

    features: Dict[str, Any] = Field(
        ...,
        description="Feature dictionary for prediction",
        examples=[{"hcp_id": "HCP001", "territory": "Northeast", "specialty": "Oncology"}],
    )
    entity_id: Optional[str] = Field(
        default=None,
        description="Entity ID for feature store lookup (if features not provided)",
    )
    time_horizon: str = Field(
        default="short_term",
        description="Prediction time horizon",
        pattern="^(short_term|medium_term|long_term)$",
    )
    return_probabilities: bool = Field(
        default=False,
        description="Return class probabilities (classification models)",
    )
    return_intervals: bool = Field(
        default=False,
        description="Return prediction intervals (regression models)",
    )
    return_feature_importance: bool = Field(
        default=False,
        description=(
            "Populate ``feature_importance`` with REAL per-prediction SHAP "
            "contributions for this exact input. Computed on the gold-standard "
            "raw-covariate path by delegating to the BentoML ``/shap`` endpoint "
            "(LinearExplainer over the routed model). Off by default so legacy "
            "callers pay no SHAP latency; best-effort (a SHAP failure does NOT "
            "fail the prediction — feature_importance is simply left null)."
        ),
    )


class PredictionResponse(BaseModel):
    """Response schema for model prediction."""

    model_name: str = Field(..., description="Name of the model used")
    prediction: Any = Field(..., description="Model prediction value")
    confidence: Optional[float] = Field(
        default=None,
        description="Prediction confidence score (0-1)",
    )
    probabilities: Optional[Dict[str, float]] = Field(
        default=None,
        description="Class probabilities (classification only)",
    )
    prediction_interval: Optional[Dict[str, float]] = Field(
        default=None,
        description="Prediction interval (regression only)",
        examples=[{"lower": 0.1, "upper": 0.9}],
    )
    feature_importance: Optional[Dict[str, float]] = Field(
        default=None,
        description="Feature importance scores for this prediction",
    )
    latency_ms: float = Field(..., description="Prediction latency in milliseconds")
    model_version: Optional[str] = Field(default=None, description="Model version used")
    timestamp: str = Field(..., description="Prediction timestamp (ISO format)")
    feature_source: Optional[str] = Field(
        default=None,
        description=(
            "Telemetry tag describing where the prediction's features came from: "
            "'feast_online' when the route fetched features from the Feast online "
            "store using request.entity_id, or 'user_provided' when the caller "
            "supplied them directly in request.features. None for paths that "
            "predate this contract."
        ),
    )


class BatchPredictionRequest(BaseModel):
    """Request schema for batch predictions."""

    instances: List[PredictionRequest] = Field(
        ...,
        description="List of prediction requests",
        min_length=1,
        max_length=1000,
    )


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""

    model_name: str = Field(..., description="Name of the model used")
    predictions: List[PredictionResponse] = Field(
        ...,
        description="List of prediction results",
    )
    total_count: int = Field(..., description="Total number of predictions")
    success_count: int = Field(..., description="Number of successful predictions")
    failed_count: int = Field(..., description="Number of failed predictions")
    total_latency_ms: float = Field(..., description="Total processing time")
    timestamp: str = Field(..., description="Batch processing timestamp")


class ModelHealthResponse(BaseModel):
    """Response schema for model health check."""

    model_name: str = Field(..., description="Name of the model")
    status: str = Field(..., description="Health status", pattern="^(healthy|unhealthy|unknown)$")
    endpoint: str = Field(..., description="Model endpoint URL")
    last_check: str = Field(..., description="Last health check timestamp")
    error: Optional[str] = Field(default=None, description="Error message if unhealthy")


class ModelsStatusResponse(BaseModel):
    """Response schema for all models status."""

    total_models: int = Field(..., description="Total number of registered models")
    healthy_count: int = Field(..., description="Number of healthy models")
    unhealthy_count: int = Field(..., description="Number of unhealthy models")
    models: List[ModelHealthResponse] = Field(..., description="Individual model statuses")
    timestamp: str = Field(..., description="Status check timestamp")


# =============================================================================
# FEATURE SOURCE TELEMETRY TAGS
# =============================================================================
#
# Feature-source tag values returned in PredictionResponse.feature_source and
# forwarded to BentoML via input_data["feature_source"]:
#   - "feast_online":  features were fetched server-side from the Feast online
#                      store using PredictionRequest.entity_id.
#   - "user_provided": features came directly from PredictionRequest.features.

FEATURE_SOURCE_FEAST_ONLINE = "feast_online"
FEATURE_SOURCE_USER_PROVIDED = "user_provided"


# Canonical model-name → Feast feature-refs registry imported from
# ``src/feature_store/model_feature_refs.py`` (3A-M-1). Both names are
# re-bound to underscore aliases to preserve the prior module-level API
# (test fixtures monkey-patch ``predictions._MODEL_FEATURE_REFS``).
__all__ = ["_MODEL_FEATURE_REFS", "_feature_refs_for_model"]


async def _resolve_feast_client() -> FeastClient:
    """Return the singleton FeastClient (test seam).

    Why this indirection layer exists:

    1. ``get_feast_client(config: Optional[FeastConfig] = None)`` cannot
       be wired as a FastAPI ``Depends(get_feast_client)`` directly —
       FastAPI would walk the parameter list and attempt to inject
       ``config`` as a query/body parameter, which would conflict with
       the route's existing ``PredictionRequest`` body and produce a
       422-level disambiguation error at request-parse time.
    2. Wrapping it as a zero-argument coroutine here makes the symbol
       monkey-patchable on the route module by tests — the standard
       ``app.dependency_overrides[get_feast_client] = ...`` pattern
       wouldn't help because the route never declares ``get_feast_client``
       as a ``Depends(...)``.

    Tests therefore monkey-patch ``predictions._resolve_feast_client``
    directly; see ``tests/api/test_predictions_endpoints.py::mock_feast_client``.
    """
    return await get_feast_client()


# =============================================================================
# PREDICTION ENDPOINTS
# =============================================================================


@router.post(
    "/predict/{model_name}",
    response_model=PredictionResponse,
    summary="Make a single prediction",
    operation_id="predict_single",
    description=(
        "Call a BentoML model endpoint for prediction. When ``entity_id`` is "
        "supplied on the request, this route fetches features from the Feast "
        "online store server-side and passes the resulting feature row to "
        "BentoML; the response is tagged ``feature_source='feast_online'``. "
        "When only ``features`` is supplied, the dict is forwarded as-is and "
        "tagged ``feature_source='user_provided'``."
    ),
)
async def predict(
    model_name: str,
    request: PredictionRequest,
    client: BentoMLClient = Depends(get_bentoml_client),
    user: Dict[str, Any] = Depends(require_auth),
) -> PredictionResponse:
    """Make a prediction using the specified model.

    Behavior matrix:
      - ``request.entity_id`` set                  → Feast online lookup;
        feature_source = 'feast_online'.
      - Only ``request.features`` set              → forward as-is;
        feature_source = 'user_provided'.
      - Both set                                   → Feast wins (matches the
        documented intent of the entity_id field).
      - Neither set                                → Pydantic 422 (features is
        required on PredictionRequest).

    Args:
        model_name: Name of the model to use
        request: Prediction request data
        client: BentoML client (injected)

    Returns:
        Prediction result with metadata, including ``feature_source``.

    Raises:
        HTTPException: If model not found or service unavailable
    """
    try:
        feature_source = FEATURE_SOURCE_USER_PROVIDED
        features_payload: Dict[str, Any] = request.features

        if request.entity_id:
            feature_refs = _feature_refs_for_model(model_name)
            try:
                feast_client = await _resolve_feast_client()
                feast_response = await feast_client.get_online_features(
                    entity_rows=[{"patient_id": request.entity_id}],
                    feature_refs=feature_refs,
                    full_feature_names=False,
                )
                # Collapse list-per-feature shape to single-row dict (one entity).
                features_payload = {k: (v[0] if v else None) for k, v in feast_response.items()}
            except Exception as e:
                logger.error(
                    "Feast online lookup failed for entity_id=%s, model=%s: %s",
                    request.entity_id,
                    model_name,
                    e,
                )
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"Feature store lookup failed: {e}",
                )

            # #576 anti-null-trap guard: a Feast 200 can carry PRESENT-but-null
            # values (verified live — a single-key patient lookup against a
            # composite-keyed view returns nulls, not an error). Labeling that
            # feature_source='feast_online' feeds the model a null vector while
            # presenting it as real — the #532 harm that no exception catches.
            # Fail loud instead of mislabeling. Placed OUTSIDE the lookup ``try``
            # so this 503 is not re-wrapped by the broad ``except Exception``.
            missing = missing_or_null_feature_fields(features_payload, feature_refs)
            if missing:
                logger.error(
                    "Feast returned null/missing required features %s for "
                    "entity_id=%s (model=%s); refusing to label "
                    "feature_source='feast_online' over an incomplete vector (#576).",
                    missing,
                    request.entity_id,
                    model_name,
                )
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Feature store returned incomplete features",
                )

            feature_source = FEATURE_SOURCE_FEAST_ONLINE
            logger.info(
                "Fetched %d Feast features for entity_id=%s (model=%s)",
                len(features_payload),
                request.entity_id,
                model_name,
            )

        # Build the BentoML input. ALL paths now carry ``model_name`` so the
        # multi-model service routes to the requested bundle. Without it the
        # service falls back to the unloaded legacy default ("no_model") and
        # returns an EMPTY ``predictions`` list — #39 routed /model_info + /shap
        # by name but left this plain /predict path unrouted, so every
        # gold-standard prediction hit "no_model" and 500'd on the empty index.
        input_data: Dict[str, Any]
        used_raw_path = False
        if request.entity_id:
            # Feast path: the online store yields a feature row that we vectorize
            # into the model's authoritative POSITIONAL order (resolved from
            # /model_info, never guessed). Fails closed on a missing required
            # feature (no silent zero-fill).
            feature_order = await _resolve_feature_order(client, model_name)
            ordered_row = _vectorize_feature_dict(
                features_payload, feature_order, context=f"predict(model={model_name})"
            )
            input_data = {
                "features": [ordered_row],
                "model_name": model_name,
                "model_type": "classification",
                "return_proba": request.return_probabilities,
                "return_intervals": request.return_intervals,
                "feature_source": feature_source,
                "entity_id": request.entity_id,
            }
        else:
            # User-provided features. Resolve the model's contract from
            # /model_info to choose the encoding path WITHOUT guessing:
            #   - gold-standard models bundle a FeatureBuilder and expose
            #     ``keep_columns`` (RAW human covariates — e.g. disease_severity,
            #     academic_hcp, geographic_region) -> forward ``raw_features`` and
            #     let the bundle one-hot-encode them server-side (a human cannot
            #     pre-engineer the positional encoded vector).
            #   - legacy/positional models expose only ``feature_columns`` -> the
            #     caller already supplies encoded numeric values, so vectorize the
            #     dict into the ordered positional row (the original contract).
            # Branching keeps BOTH families correct: a FeatureBuilder model fed a
            # raw dict it cannot encode, or a positional model fed raw_features,
            # would otherwise 500.
            try:
                model_info = await client.get_model_info(model_name)
            except Exception as e:
                logger.error("Could not fetch model_info for predict (model=%s): %s", model_name, e)
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"Model metadata unavailable for '{model_name}'",
                )

            keep_columns = model_info.get("keep_columns")
            if isinstance(keep_columns, list) and keep_columns:
                # Gold-standard raw-covariate path. Validate completeness HERE for
                # an honest 422 — the served FeatureBuilder rejects an incomplete
                # row with an opaque 500, so we name the missing covariates instead.
                missing = [
                    c
                    for c in keep_columns
                    if c not in features_payload or features_payload[c] is None
                ]
                if missing:
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail=(
                            f"Missing required covariate(s) for '{model_name}': {missing}. "
                            f"Expected raw covariates: {list(keep_columns)}"
                        ),
                    )
                used_raw_path = True
                input_data = {
                    "raw_features": [features_payload],
                    "model_name": model_name,
                    "model_type": "classification",
                    "return_proba": request.return_probabilities,
                    "return_intervals": request.return_intervals,
                    "feature_source": feature_source,
                }
            else:
                # Legacy positional path (original contract). The dict carries
                # already-encoded numeric values; vectorize into feature_columns
                # order, failing closed on a missing column or an absent order
                # (mirrors ``_resolve_feature_order`` using the fetched metadata).
                columns = model_info.get("feature_columns")
                if not columns or not isinstance(columns, list):
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail=(
                            f"Model '{model_name}' does not expose a feature order; "
                            "cannot vectorize feature dictionary"
                        ),
                    )
                feature_order = [str(c) for c in columns]
                ordered_row = _vectorize_feature_dict(
                    features_payload, feature_order, context=f"predict(model={model_name})"
                )
                input_data = {
                    "features": [ordered_row],
                    "model_name": model_name,
                    "model_type": "classification",
                    "return_proba": request.return_probabilities,
                    "return_intervals": request.return_intervals,
                    "feature_source": feature_source,
                }

        # Call BentoML endpoint
        result = await client.predict(model_name, input_data)

        # Fail closed on a service-reported error (the #39 contract returns an
        # unknown ``model_name`` as a 200 with a non-null ``error`` + empty
        # predictions). Surface it instead of indexing the empty list.
        service_error = result.get("error")
        if service_error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Model '{model_name}': {service_error}",
            )

        # Resolve the scalar prediction WITHOUT an unguarded ``[0]``. The prior
        # ``result.get("predictions", [None])[0]`` raised "list index out of
        # range" (opaque 500) whenever the service returned an empty list — the
        # exact symptom of the unrouted-model bug above. A legitimately falsy
        # prediction (0, 0.0, False) must still pass through, so the fallback is
        # gated on ``is None``, not truthiness.
        raw_prediction = result.get("prediction")
        if raw_prediction is None:
            preds = result.get("predictions") or []
            raw_prediction = preds[0] if preds else None
        if raw_prediction is None:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Model '{model_name}' returned no prediction",
            )

        # Extract metadata
        metadata = result.get("_metadata", {})

        # The live flat contract returns ``probabilities`` as a flat list
        # (positive-class probability per sample). PredictionResponse expects a
        # labeled dict, so map the single-sample positive-class probability to
        # {"positive_class": p} and surface it as ``confidence`` too. A legacy
        # dict response (older/mock servers) passes through unchanged.
        raw_probs = result.get("probabilities")
        probabilities: Optional[Dict[str, float]]
        confidence = result.get("confidence")
        if isinstance(raw_probs, dict):
            probabilities = raw_probs
        elif isinstance(raw_probs, list) and raw_probs:
            positive = float(raw_probs[0])
            probabilities = {"positive_class": positive}
            if confidence is None:
                confidence = positive
        else:
            probabilities = None

        # Real per-prediction feature contributions (opt-in). The /predict
        # contract carries no feature_importance, so when the caller asks for it
        # on the raw-covariate path we delegate to the BentoML ``/shap`` endpoint
        # (LinearExplainer over the routed model's inner LR) for the SAME raw row.
        # Best-effort: a SHAP failure must NOT fail the prediction — we log and
        # leave feature_importance null (the UI hides the contributions block).
        feature_importance = result.get("feature_importance")
        if request.return_feature_importance and used_raw_path and feature_importance is None:
            try:
                shap_result = await client.get_shap(model_name, [features_payload])
                if not shap_result.get("error"):
                    shap_values = shap_result.get("shap_values") or {}
                    # Drop display-noise: encoded one-hots with no contribution
                    # for this row (|shap| ~ 0) would render as "+0.0%" bars.
                    feature_importance = {
                        name: float(value)
                        for name, value in shap_values.items()
                        if abs(float(value)) >= 1e-6
                    } or None
            except Exception as shap_exc:  # noqa: BLE001 — contributions are best-effort
                logger.warning(
                    "SHAP feature_importance unavailable for model=%s (non-fatal): %s",
                    model_name,
                    shap_exc,
                )

        # 3A-I-3: route is the source of truth for feature_source.
        # If we invoked Feast (entity_id was set + lookup succeeded), the
        # user request was Feast-driven regardless of what BentoML reports —
        # BentoML may legitimately report a fallback path on its end, but
        # from the route's perspective the *request* was a Feast lookup.
        # Conversely, if no entity_id was supplied we used the caller's
        # raw features and must not be overridden into 'feast_online' by
        # a downstream value.
        return PredictionResponse(
            model_name=model_name,
            prediction=raw_prediction,
            confidence=confidence,
            probabilities=probabilities,
            prediction_interval=result.get("prediction_interval"),
            feature_importance=feature_importance,
            latency_ms=metadata.get("latency_ms", 0),
            # Live contract carries ``model_id``; legacy mocks carry ``model_version``.
            model_version=result.get("model_version") or result.get("model_id"),
            timestamp=metadata.get("timestamp", datetime.now(timezone.utc).isoformat()),
            feature_source=feature_source,
        )

    except HTTPException:
        # Already-shaped HTTP errors (e.g. Feast 503) pass through unchanged.
        raise
    except RuntimeError as e:
        # Circuit breaker open
        logger.warning(f"Model service unavailable: {model_name} - {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Prediction failed for model {model_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


@router.post(
    "/predict/{model_name}/batch",
    response_model=BatchPredictionResponse,
    summary="Make batch predictions",
    operation_id="predict_batch",
    description="Call a BentoML model endpoint for multiple predictions",
)
async def predict_batch(
    model_name: str,
    request: BatchPredictionRequest,
    client: BentoMLClient = Depends(get_bentoml_client),
    user: Dict[str, Any] = Depends(require_auth),
) -> BatchPredictionResponse:
    """Make batch predictions using the specified model.

    Args:
        model_name: Name of the model to use
        request: Batch prediction request data
        client: BentoML client (injected)

    Returns:
        Batch prediction results with metadata
    """
    import time

    start_time = time.time()
    predictions = []
    success_count = 0
    failed_count = 0

    try:
        # Build batch input matching the BentoML ``BatchPredictionInput`` schema
        # (verified live 2026-06-14): {"batch_id": str, "features": [[...], ...]}.
        # Each instance's feature DICT is vectorized into the model's authoritative
        # POSITIONAL order (resolved from /model_info, never guessed). A missing
        # required feature on any instance fails closed (422) — no zero-fill.
        import uuid

        feature_order = await _resolve_feature_order(client, model_name)
        ordered_rows = [
            _vectorize_feature_dict(
                inst.features,
                feature_order,
                context=f"predict_batch(model={model_name}, instance={i})",
            )
            for i, inst in enumerate(request.instances)
        ]

        batch_data = {
            "batch_id": str(uuid.uuid4()),
            "features": ordered_rows,
        }

        # Call batch endpoint
        result = await client.predict_batch(model_name, batch_data)

        # Flat-contract response: {"batch_id", "total_samples", "predictions":
        # [number, ...], "processing_time_ms", "is_mock"}. ``predictions`` is a
        # flat list of scalar predictions (one per instance), NOT a list of
        # per-instance result dicts.
        raw_predictions = result.get("predictions", [])
        model_version = result.get("model_id")
        for pred in raw_predictions:
            predictions.append(
                PredictionResponse(
                    model_name=model_name,
                    prediction=pred,
                    confidence=None,
                    probabilities=None,
                    prediction_interval=None,
                    feature_importance=None,
                    latency_ms=result.get("processing_time_ms", 0),
                    model_version=model_version,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
            )
            success_count += 1

    except HTTPException:
        # Preserve the fail-closed contract: _resolve_feature_order raises 503
        # (no feature order) and _vectorize_feature_dict raises 422 (missing /
        # non-numeric required feature). These must NOT be rewritten to 500.
        raise
    except Exception as e:
        logger.error(f"Batch prediction failed for model {model_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}",
        )

    total_latency = (time.time() - start_time) * 1000

    return BatchPredictionResponse(
        model_name=model_name,
        predictions=predictions,
        total_count=len(request.instances),
        success_count=success_count,
        failed_count=failed_count,
        total_latency_ms=total_latency,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# =============================================================================
# HEALTH & STATUS ENDPOINTS
# =============================================================================


@router.get(
    "/{model_name}/health",
    response_model=ModelHealthResponse,
    summary="Check model health",
    operation_id="get_prediction_model_health",
    description="Check the health status of a specific BentoML model service",
)
async def model_health(
    model_name: str,
    client: BentoMLClient = Depends(get_bentoml_client),
) -> ModelHealthResponse:
    """Check health of a specific model service.

    Args:
        model_name: Name of the model to check
        client: BentoML client (injected)

    Returns:
        Model health status
    """
    result = await client.health_check(model_name)

    return ModelHealthResponse(
        model_name=model_name,
        status=result.get("status", "unknown"),
        endpoint=result.get("endpoint", ""),
        last_check=result.get("timestamp", datetime.now(timezone.utc).isoformat()),
        error=result.get("error"),
    )


# ---------------------------------------------------------------------------
# Curated, brand/cohort-appropriate model input schema (predictive-analytics)
# ---------------------------------------------------------------------------
# The gold-standard models expose only their RAW encoded ``feature_columns``, so
# a form derived from them surfaced the GENERIC specialty pool for every brand —
# e.g. the Kisqali (HR+/HER2- breast cancer) model offered CSU specialties
# (dermatology, allergy_immunology). We curate the input fields from the SAME
# canonical sources the synthetic data was generated from: the cohort covariates
# (``cohort_spec``) + brand-appropriate specialty choices
# (``HCPGenerator.BRAND_SPECIALTY_DIST``). Returns None for any model whose
# brand/cohort cannot be resolved, so the frontend falls back to its generic path.

_CURATED_BRANDS = {
    "kisqali": "Kisqali",
    "fabhalta": "Fabhalta",
    "remibrutinib": "Remibrutinib",
}
_PATIENT_COHORTS = ("initiation", "persistence", "discontinuation")
_HCP_COHORT = "hcp_adoption"
# Mirrors RegionEnum / HCPGenerator.REGION_DIST (brand-agnostic US regions).
_REGION_CHOICES = ["northeast", "south", "midwest", "west"]
# T9: insurance_type is a categorical persistence driver (access gradient) — render
# it as a dropdown on the score-cohort form, not a free numeric input.
_INSURANCE_CHOICES = ["commercial", "medicare", "medicaid"]


def _parse_model_brand_cohort(model_name: str):
    """Resolve ``(cohort, brand)`` from a ``<cohort>_<brand>_goldstd_lr_v1`` name."""
    low = model_name.lower()
    brand = next((b for b in _CURATED_BRANDS if b in low), None)
    if low.startswith(_HCP_COHORT):
        cohort = _HCP_COHORT
    else:
        cohort = next((c for c in _PATIENT_COHORTS if low.startswith(c)), None)
    return cohort, brand


def build_curated_input_fields(model_name: str) -> Optional[List[Dict[str, Any]]]:
    """Brand/cohort-appropriate input fields for a gold-standard model, or None.

    Each entry is ``{name, type, choices?}`` over the model's REAL covariates
    (``cohort_spec`` base_covariates), with brand-appropriate categorical choices
    so the predictive-analytics form is clinically coherent (Kisqali -> oncology,
    Remibrutinib -> CSU specialties, Fabhalta -> PNH specialties). Numeric
    covariates stay ``type="number"``; values map 1:1 to the model's inputs, so
    prediction is unaffected. ``None`` (unresolved model) lets the FE fall back.
    """
    cohort, brand = _parse_model_brand_cohort(model_name)
    if not cohort or not brand:
        return None
    brand_proper = _CURATED_BRANDS[brand]
    try:
        from src.mlops.gold_standard_eval.cohort_spec import (
            make_hcp_spec,
            make_patient_spec,
        )

        if cohort == _HCP_COHORT:
            covariates = make_hcp_spec(brand_proper).base_covariates
        else:
            covariates = make_patient_spec(cohort, brand_proper).base_covariates
    except Exception:  # unknown cohort/brand -> let the FE fall back
        return None

    specialty_choices: List[str] = []
    try:
        from src.ml.synthetic.config import Brand
        from src.ml.synthetic.generators.hcp_generator import HCPGenerator

        dist = HCPGenerator.BRAND_SPECIALTY_DIST.get(Brand(brand_proper), {})
        specialty_choices = [s.value for s in dist]
    except Exception:
        specialty_choices = []

    fields: List[Dict[str, Any]] = []
    for cov in covariates:
        if cov == "specialty" and specialty_choices:
            fields.append({"name": cov, "type": "category", "choices": specialty_choices})
        elif cov == "geographic_region":
            fields.append({"name": cov, "type": "category", "choices": list(_REGION_CHOICES)})
        elif cov == "insurance_type":
            fields.append({"name": cov, "type": "category", "choices": list(_INSURANCE_CHOICES)})
        else:
            fields.append({"name": cov, "type": "number"})
    return fields


@router.get(
    "/{model_name}/info",
    summary="Get model metadata",
    operation_id="get_model_info",
    description="Get metadata and configuration for a deployed model",
)
async def model_info(
    model_name: str,
    client: BentoMLClient = Depends(get_bentoml_client),
) -> Dict[str, Any]:
    """Get metadata for a specific model.

    Augments the BentoML metadata with a curated, brand/cohort-appropriate
    ``input_fields`` schema (see :func:`build_curated_input_fields`) so the
    predictive-analytics form offers clinically coherent features and choices.

    Args:
        model_name: Name of the model
        client: BentoML client (injected)

    Returns:
        Model metadata and configuration
    """
    try:
        info = await client.get_model_info(model_name)
        curated = build_curated_input_fields(model_name)
        if curated:
            info = {**info, "input_fields": curated}
        return info
    except Exception as e:
        logger.error(f"Failed to get info for model {model_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model not found or unavailable: {model_name}",
        )


@router.get(
    "/status",
    response_model=ModelsStatusResponse,
    summary="Get all models status",
    operation_id="get_models_status",
    description="Get health status of all registered BentoML model services",
)
async def models_status(
    client: BentoMLClient = Depends(get_bentoml_client),
    models: Optional[List[str]] = Query(
        default=None,
        description="Specific models to check. If not provided, checks all registered models.",
    ),
) -> ModelsStatusResponse:
    """Get status of all registered models.

    Args:
        client: BentoML client (injected)
        models: Optional list of specific models to check

    Returns:
        Status of all models
    """
    # Drive the model list from the registry, NOT fictional hardcoded handles.
    # When the caller does not name specific models, resolve the real
    # production models from ``ml_model_registry`` (stage='production',
    # is_synthetic=false). An empty result yields an honest empty status rather
    # than fabricated handles that never resolve.
    model_list = models or await _resolve_production_model_names()

    model_statuses = []
    healthy_count = 0
    unhealthy_count = 0

    for model_name in model_list:
        result = await client.health_check(model_name)
        status_str = result.get("status", "unknown")

        if status_str == "healthy":
            healthy_count += 1
        else:
            unhealthy_count += 1

        model_statuses.append(
            ModelHealthResponse(
                model_name=model_name,
                status=status_str,
                endpoint=result.get("endpoint", ""),
                last_check=result.get("timestamp", datetime.now(timezone.utc).isoformat()),
                error=result.get("error"),
            )
        )

    return ModelsStatusResponse(
        total_models=len(model_list),
        healthy_count=healthy_count,
        unhealthy_count=unhealthy_count,
        models=model_statuses,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# =============================================================================
# COHORT SCORING — data-driven population view (replaces hand-typed features)
# =============================================================================
#
# Instead of asking the user to type one feature row, score the model's OWN real
# holdout cohort (out-of-sample) and return a RANKED list of targets + the
# probability distribution. The row source is the model's own
# ``FeatureBuilder.load_frame(splits=["holdout"])`` (same training schema, both
# grains, real entity ids); scoring goes through the BentoML raw-covariate BATCH
# path in chunks (patient holdout ~5k rows/brand). SHAP stays O(top-N), never
# O(cohort): cohort-level drivers sample only the top-ranked rows
# (_DRIVER_SHAP_ROWS); full per-row SHAP remains the single-predict drill-down's
# job. Synthetic-for-now (include_real defaults False); labeled honestly.


class CohortScoredRow(BaseModel):
    """One scored entity from the holdout cohort."""

    entity_id: str = Field(
        ..., description="Real entity id (patient_id / hcp_id) from the holdout split"
    )
    probability: float = Field(..., description="Predicted positive-class probability")
    covariates: Dict[str, Any] = Field(
        default_factory=dict, description="The RAW covariates scored for this entity"
    )


class CohortDriver(BaseModel):
    """One cohort-level SHAP driver: mean |SHAP| over the sampled top targets."""

    feature: str = Field(..., description="Encoded feature name (matches drill-down SHAP)")
    importance: float = Field(
        ..., description="Mean |SHAP| across the sampled top-ranked rows (log-odds scale)"
    )
    direction: str = Field(
        ...,
        description=(
            "Sign of the mean signed SHAP across sampled rows: 'increases' | "
            "'decreases' | 'mixed' (mean ~0 while |SHAP| is not)"
        ),
    )


class CohortScoreDistribution(BaseModel):
    """Probability distribution over ALL scored rows (not just the top-N)."""

    n: int = Field(..., description="Number of rows scored")
    mean: float = Field(..., description="Mean predicted probability")
    bin_edges: List[float] = Field(
        default_factory=list, description="Histogram bin edges (len = bins + 1)"
    )
    bin_counts: List[int] = Field(
        default_factory=list, description="Row count per [0,1] probability bin"
    )


class CohortScoreResponse(BaseModel):
    """Async cohort-scoring job: submit -> poll. Ranked targets + distribution."""

    job_id: str
    status: str = Field(..., description="pending | running | completed | failed")
    model_name: str
    cohort: Optional[str] = Field(
        default=None, description="Resolved cohort (e.g. initiation, hcp_adoption)"
    )
    brand: Optional[str] = Field(default=None, description="Resolved brand (e.g. Kisqali)")
    split: str = Field(default="holdout", description="Data split scored (out-of-sample)")
    out_of_sample: bool = Field(
        default=True, description="True — the holdout split is held out of training"
    )
    feature_source: str = Field(
        default="holdout_synthetic",
        description="Provenance: synthetic holdout cohort (real rows when include_real flips). Honest labeling.",
    )
    n_scored: int = Field(default=0, description="Rows scored so far / total")
    top_n: int = Field(default=0, description="Number of top-ranked rows returned")
    top_rows: List[CohortScoredRow] = Field(
        default_factory=list,
        description="Highest-probability entities (ranked desc), capped at top_n",
    )
    distribution: Optional[CohortScoreDistribution] = Field(
        default=None, description="Probability distribution over all scored rows"
    )
    top_drivers: List[CohortDriver] = Field(
        default_factory=list,
        description=(
            "Cohort-level SHAP drivers: mean |SHAP| per encoded feature over the "
            "top-ranked rows (capped at drivers_from_top_n), desc. Best-effort — "
            "empty when SHAP is unavailable, never fabricated."
        ),
    )
    drivers_from_top_n: int = Field(
        default=0,
        description="How many top-ranked rows the driver aggregation actually sampled",
    )
    error: Optional[str] = Field(default=None, description="Set when status == failed")
    latency_ms: float = Field(default=0.0, description="Scoring wall-clock once completed")


_COHORT_SCORE_BINS = 10

# Cohort-level drivers sample the top-ranked rows only: one serving /shap call
# per row (~5ms measured live), so 50 rows adds well under a second to the async
# job while staying O(top-N), never O(cohort).
_DRIVER_SHAP_ROWS = 50
_DRIVER_TOP_K = 10


def aggregate_shap_drivers(
    shap_rows: List[Dict[str, float]], top_k: int = _DRIVER_TOP_K
) -> List[CohortDriver]:
    """Pure: mean |SHAP| (and mean signed SHAP for direction) per encoded feature
    over the sampled rows, ranked desc by mean |SHAP|, capped at ``top_k``.
    Features contributing ~nothing everywhere (mean |SHAP| < 1e-6) are dropped."""
    if not shap_rows:
        return []
    abs_sums: Dict[str, float] = {}
    signed_sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for row in shap_rows:
        for feature, value in row.items():
            v = float(value)
            abs_sums[feature] = abs_sums.get(feature, 0.0) + abs(v)
            signed_sums[feature] = signed_sums.get(feature, 0.0) + v
            counts[feature] = counts.get(feature, 0) + 1
    drivers = []
    for feature, abs_sum in abs_sums.items():
        n = counts[feature]
        mean_abs = abs_sum / n
        if mean_abs < 1e-6:
            continue
        mean_signed = signed_sums[feature] / n
        # 'mixed': per-row contributions cancel out (|mean| well below mean |.|),
        # so calling it increases/decreases would overstate a consistent direction.
        if abs(mean_signed) < 0.25 * mean_abs:
            direction = "mixed"
        else:
            direction = "increases" if mean_signed > 0 else "decreases"
        drivers.append(CohortDriver(feature=feature, importance=mean_abs, direction=direction))
    drivers.sort(key=lambda d: d.importance, reverse=True)
    return drivers[: max(0, top_k)]


async def _sample_cohort_drivers(
    client: Any, model_name: str, top_rows: List["CohortScoredRow"]
) -> tuple[List[CohortDriver], int]:
    """Best-effort (drivers, n_rows_sampled): per-row serving /shap over the top
    rows, aggregated by ``aggregate_shap_drivers``. Any failure yields ([], 0) —
    the cohort job must never fail (or fabricate drivers) because of SHAP."""
    sample = top_rows[:_DRIVER_SHAP_ROWS]
    shap_rows: List[Dict[str, float]] = []
    try:
        for row in sample:
            result = await client.get_shap(model_name, [row.covariates])
            if result.get("error"):
                continue
            values = result.get("shap_values") or {}
            if values:
                shap_rows.append({str(k): float(v) for k, v in values.items()})
    except Exception as exc:  # noqa: BLE001 — drivers are best-effort
        logger.warning(
            "cohort-level SHAP drivers unavailable for model=%s (non-fatal): %s",
            model_name,
            exc,
        )
        return [], 0
    return aggregate_shap_drivers(shap_rows), len(shap_rows)


def _cohort_ranking(
    entity_ids: List[str],
    covariate_rows: List[Dict[str, Any]],
    probabilities: List[float],
    top_n: int,
):
    """Pure: rank rows by probability desc (top-N) and summarize the distribution
    over ALL rows into fixed [0,1] bins. No I/O — the unit-testable core."""
    n = len(probabilities)
    edges = [i / _COHORT_SCORE_BINS for i in range(_COHORT_SCORE_BINS + 1)]
    counts = [0] * _COHORT_SCORE_BINS
    for p in probabilities:
        idx = int(p * _COHORT_SCORE_BINS)
        if idx >= _COHORT_SCORE_BINS:  # 1.0 lands in the last bin (inclusive)
            idx = _COHORT_SCORE_BINS - 1
        elif idx < 0:
            idx = 0
        counts[idx] += 1
    mean = (sum(probabilities) / n) if n else 0.0
    dist = CohortScoreDistribution(n=n, mean=mean, bin_edges=edges, bin_counts=counts)
    rows = [
        CohortScoredRow(entity_id=str(eid), probability=float(p), covariates=cov)
        for eid, cov, p in zip(entity_ids, covariate_rows, probabilities, strict=True)
    ]
    rows.sort(key=lambda r: r.probability, reverse=True)
    return rows[: max(0, top_n)], dist


async def _score_cohort_chunks(
    client: Any, model_name: str, raw_features: List[Dict[str, Any]], chunk_size: int = 1000
) -> List[float]:
    """Score raw covariate rows through the BentoML raw-covariate BATCH path in
    chunks of ``chunk_size``. Fails closed on a service error or a per-chunk
    probability/row length mismatch (never zero-fills a missing score)."""
    probabilities: List[float] = []
    for start in range(0, len(raw_features), chunk_size):
        chunk = raw_features[start : start + chunk_size]
        try:
            result = await client.predict_batch(
                model_name,
                {"batch_id": str(uuid.uuid4()), "raw_features": chunk, "model_name": model_name},
            )
        except httpx.HTTPStatusError as e:
            # The BentoML client (bentoml_client.predict_batch) calls
            # response.raise_for_status() and re-raises, so a non-2xx never reaches
            # the `result.get("error")` branch below. A 400/422 here means the live
            # serving schema rejected the current raw-covariate batch shape — i.e.
            # the deployed bentoml service is STALE relative to
            # scripts/bentoml/e2i_serving_service.py. Translate it into an actionable
            # 502 instead of letting a bare httpx error surface as the job cause.
            code = e.response.status_code
            if code in (400, 422):
                detail = (
                    f"Cohort scoring rejected by the model server (HTTP {code}) for "
                    f"'{model_name}': the BentoML serving schema is stale and no longer "
                    "matches the request shape. Restart the bentoml service so it reloads "
                    "the current scripts/bentoml/e2i_serving_service.py."
                )
            else:
                detail = (
                    f"Model server returned HTTP {code} scoring cohort '{model_name}'. "
                    "Check the bentoml service logs."
                )
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=detail) from e
        except httpx.RequestError as e:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=(
                    f"Could not reach the model server to score '{model_name}': {e}. "
                    "Is the bentoml service running?"
                ),
            ) from e
        err = result.get("error")
        if err:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Cohort scoring failed for '{model_name}': {err}",
            )
        chunk_probs = result.get("probabilities") or []
        if len(chunk_probs) != len(chunk):
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=(
                    f"Model '{model_name}' returned {len(chunk_probs)} probabilities "
                    f"for {len(chunk)} rows"
                ),
            )
        probabilities.extend(float(p) for p in chunk_probs)
    return probabilities


def _resolve_cohort_spec(model_name: str):
    """Resolve ``(cohort, brand_proper, spec, entity_col)`` from a gold-standard
    model name, or raise ValueError when it is not a resolvable cohort model."""
    cohort, brand = _parse_model_brand_cohort(model_name)
    if not cohort or not brand:
        raise ValueError(
            f"'{model_name}' is not a resolvable gold-standard cohort model "
            "(expected <cohort>_<brand>_goldstd_lr_v1)"
        )
    brand_proper = _CURATED_BRANDS[brand]
    from src.mlops.gold_standard_eval.cohort_spec import make_hcp_spec, make_patient_spec

    if cohort == _HCP_COHORT:
        spec = make_hcp_spec(brand_proper)
        entity_col = "hcp_id"
    else:
        spec = make_patient_spec(cohort, brand_proper)
        entity_col = "patient_id"
    return cohort, brand_proper, spec, entity_col


def _native(value: Any) -> Any:
    """Convert a numpy scalar to a native Python scalar so it is JSON-serializable
    (the bentoml client posts ``json=``, which cannot encode numpy types)."""
    return value.item() if hasattr(value, "item") else value


_cohort_score_store: "DurableJobStore[CohortScoreResponse]" = DurableJobStore(
    "predictions:cohort_score", CohortScoreResponse
)


async def _resolve_db_client() -> Any:
    """Return the async Supabase client (lazy + test seam).

    Imported lazily so merely importing this route module does NOT pull in the
    heavy ``src.memory.services.factories`` graph (it runs service checks and
    loads embedding/redis deps at import — a real memory cost). Mirrors
    ``_resolve_feast_client``: tests monkeypatch ``predictions._resolve_db_client``
    directly rather than the factory.
    """
    from src.memory.services.factories import get_async_supabase_client

    return await get_async_supabase_client()


async def _run_cohort_score_task(
    job_id: str, model_name: str, top_n: int, client: BentoMLClient
) -> None:
    """Background: load the model's holdout cohort, score it in chunks via the raw
    BATCH path, rank by probability, and publish the completed (or failed) job."""
    import time as _time

    started = _time.time()
    try:
        cohort, brand_proper, spec, entity_col = _resolve_cohort_spec(model_name)
        from src.mlops.gold_standard_eval.feature_builder import FeatureBuilder

        fb = FeatureBuilder(spec)
        db = await _resolve_db_client()
        frame = await fb.load_frame(db, splits=["holdout"])

        keep = list(fb.keep_columns)
        if entity_col not in frame.columns:
            entity_col = next((c for c in frame.columns if c.endswith("_id")), entity_col)
        records = frame.to_dict("records")
        raw_features = [{k: _native(rec[k]) for k in keep} for rec in records]
        entity_ids = [str(_native(rec.get(entity_col, i))) for i, rec in enumerate(records)]

        await _cohort_score_store.set(
            job_id,
            CohortScoreResponse(
                job_id=job_id,
                status="running",
                model_name=model_name,
                cohort=cohort,
                brand=brand_proper,
                n_scored=0,
                top_n=top_n,
            ),
        )

        probabilities = await _score_cohort_chunks(client, model_name, raw_features)
        top_rows, dist = _cohort_ranking(entity_ids, raw_features, probabilities, top_n)
        top_drivers, drivers_from_top_n = await _sample_cohort_drivers(client, model_name, top_rows)

        await _cohort_score_store.set(
            job_id,
            CohortScoreResponse(
                job_id=job_id,
                status="completed",
                model_name=model_name,
                cohort=cohort,
                brand=brand_proper,
                n_scored=len(probabilities),
                top_n=top_n,
                top_rows=top_rows,
                distribution=dist,
                top_drivers=top_drivers,
                drivers_from_top_n=drivers_from_top_n,
                latency_ms=(_time.time() - started) * 1000,
            ),
        )
    except Exception as e:  # noqa: BLE001 — fail the job, never fabricate scores
        detail = e.detail if isinstance(e, HTTPException) else str(e)
        logger.error(
            "cohort-score %s failed: %s",
            model_name,
            detail,
            exc_info=not isinstance(e, HTTPException),
        )
        await _cohort_score_store.set(
            job_id,
            CohortScoreResponse(
                job_id=job_id,
                status="failed",
                model_name=model_name,
                error=str(detail),
                latency_ms=(_time.time() - started) * 1000,
            ),
        )


@router.post(
    "/predict/{model_name}/cohort",
    response_model=CohortScoreResponse,
    summary="Score a model's holdout cohort and rank targets (async submit -> poll)",
    operation_id="score_model_cohort",
    description=(
        "Score the model's OWN out-of-sample holdout cohort (loaded via the model's "
        "FeatureBuilder) through the raw-covariate batch path, ranked by predicted "
        "probability. Heavy (thousands of rows) -> async: returns a pending job; poll "
        "GET /predict/{model_name}/cohort/{job_id}. Replaces hand-typed input features."
    ),
)
async def score_cohort(
    model_name: str,
    background_tasks: BackgroundTasks,
    top_n: int = Query(100, ge=1, le=1000, description="Number of top-ranked entities to return"),
    client: BentoMLClient = Depends(get_bentoml_client),
    user: Dict[str, Any] = Depends(require_auth),
) -> CohortScoreResponse:
    try:
        cohort, brand_proper, _spec, _entity_col = _resolve_cohort_spec(model_name)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))

    job_id = str(uuid.uuid4())
    initial = CohortScoreResponse(
        job_id=job_id,
        status="pending",
        model_name=model_name,
        cohort=cohort,
        brand=brand_proper,
        top_n=top_n,
    )
    await _cohort_score_store.set(job_id, initial)
    background_tasks.add_task(_run_cohort_score_task, job_id, model_name, top_n, client)
    return initial


@router.get(
    "/predict/{model_name}/cohort/{job_id}",
    response_model=CohortScoreResponse,
    summary="Poll a cohort-scoring job",
    operation_id="get_model_cohort_score",
)
async def get_cohort_score(
    model_name: str,
    job_id: str,
    user: Dict[str, Any] = Depends(require_auth),
) -> CohortScoreResponse:
    job = await _cohort_score_store.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown cohort-score job '{job_id}'",
        )
    return job
