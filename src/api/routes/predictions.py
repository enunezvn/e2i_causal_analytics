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

logger = logging.getLogger(__name__)

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
# PREDICTION ENDPOINTS
# =============================================================================


@router.post(
    "/predict/{model_name}",
    response_model=PredictionResponse,
    summary="Make a single prediction",
    operation_id="predict_single",
    description="Call a BentoML model endpoint for prediction",
)
async def predict(
    model_name: str,
    request: PredictionRequest,
    client: BentoMLClient = Depends(get_bentoml_client),
    user: Dict[str, Any] = Depends(require_auth),
) -> PredictionResponse:
    """Make a prediction using the specified model.

    Args:
        model_name: Name of the model to use
        request: Prediction request data
        client: BentoML client (injected)

    Returns:
        Prediction result with metadata

    Raises:
        HTTPException: If model not found or service unavailable
    """
    try:
        # Build input data for BentoML
        input_data = {
            "features": request.features,
            "return_proba": request.return_probabilities,
            "return_intervals": request.return_intervals,
        }

        if request.entity_id:
            input_data["entity_id"] = request.entity_id

        # Call BentoML endpoint
        result = await client.predict(model_name, input_data)

        # Extract metadata
        metadata = result.get("_metadata", {})

        return PredictionResponse(
            model_name=model_name,
            prediction=result.get("prediction") or result.get("predictions", [None])[0],
            confidence=result.get("confidence"),
            probabilities=result.get("probabilities"),
            prediction_interval=result.get("prediction_interval"),
            feature_importance=result.get("feature_importance"),
            latency_ms=metadata.get("latency_ms", 0),
            model_version=result.get("model_version"),
            timestamp=metadata.get("timestamp", datetime.now(timezone.utc).isoformat()),
        )

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
        # Build batch input
        batch_data = [
            {
                "features": inst.features,
                "return_proba": inst.return_probabilities,
                "return_intervals": inst.return_intervals,
            }
            for inst in request.instances
        ]

        # Call batch endpoint
        result = await client.predict_batch(model_name, batch_data)

        # Process results
        raw_predictions = result.get("predictions", [])
        for _i, pred in enumerate(raw_predictions):
            if pred.get("error"):
                failed_count += 1
                continue

            predictions.append(
                PredictionResponse(
                    model_name=model_name,
                    prediction=pred.get("prediction"),
                    confidence=pred.get("confidence"),
                    probabilities=pred.get("probabilities"),
                    prediction_interval=pred.get("prediction_interval"),
                    feature_importance=pred.get("feature_importance"),
                    latency_ms=pred.get("latency_ms", 0),
                    model_version=pred.get("model_version"),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
            )
            success_count += 1

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
    # Default models if not specified
    model_list = models or ["churn_model", "conversion_model", "causal_model"]

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
