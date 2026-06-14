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
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from src.api.dependencies.auth import require_auth
from src.api.dependencies.bentoml_client import BentoMLClient, get_bentoml_client
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
    which are not registered and never resolve). Returns the names of models at
    ``stage='production'`` and ``is_synthetic=false`` — the genuine,
    user-facing production models (e.g. ``csu_treatment_initiation_lr_*_v1``).

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
            .eq("stage", "production")
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

        # Vectorize the feature dict into the model's authoritative POSITIONAL
        # order. The live service expects ``features`` as a 2D numeric matrix
        # (samples x features) ordered by the model's own feature_columns —
        # resolved from /model_info, never guessed. Fails closed on a missing
        # required feature (no silent zero-fill).
        feature_order = await _resolve_feature_order(client, model_name)
        ordered_row = _vectorize_feature_dict(
            features_payload, feature_order, context=f"predict(model={model_name})"
        )

        # Build input data for BentoML (flat single-model contract, verified
        # live): {"input_data": {"features": [[...]], "model_type": ...}}.
        input_data: Dict[str, Any] = {
            "features": [ordered_row],
            "model_type": "classification",
            "return_proba": request.return_probabilities,
            "return_intervals": request.return_intervals,
            "feature_source": feature_source,
        }

        if request.entity_id:
            input_data["entity_id"] = request.entity_id

        # Call BentoML endpoint
        result = await client.predict(model_name, input_data)

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
            prediction=(
                # Explicit None check: a legitimate falsy prediction (0, 0.0,
                # False — e.g. binary class 0 or a regressor emitting exactly
                # 0.0) must NOT be dropped by an ``or`` short-circuit. Only fall
                # back to ``predictions[0]`` when ``prediction`` is truly absent.
                result.get("prediction")
                if result.get("prediction") is not None
                else result.get("predictions", [None])[0]
            ),
            confidence=confidence,
            probabilities=probabilities,
            prediction_interval=result.get("prediction_interval"),
            feature_importance=result.get("feature_importance"),
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

    Args:
        model_name: Name of the model
        client: BentoML client (injected)

    Returns:
        Model metadata and configuration
    """
    try:
        return await client.get_model_info(model_name)
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
