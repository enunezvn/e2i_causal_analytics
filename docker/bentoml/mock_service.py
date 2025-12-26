"""Mock BentoML Service for Testing.

This mock service provides the same API endpoints as the production
services but returns mock predictions. Used for:
- Docker infrastructure testing
- E2E test validation
- Development without real models

Version: 1.0.0
"""

import logging
import random
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

import bentoml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ============================================================================
# Request/Response Models
# ============================================================================


class PredictionInput(BaseModel):
    """Input schema for prediction requests."""

    features: List[List[float]] = Field(
        ...,
        description="Feature matrix (samples x features)",
    )
    model_type: str = Field(
        default="classification",
        description="Type of prediction: classification or regression",
    )


class PredictionOutput(BaseModel):
    """Output schema for prediction responses."""

    predictions: List[float] = Field(
        ...,
        description="Model predictions",
    )
    probabilities: List[float] = Field(
        default_factory=list,
        description="Prediction probabilities (classification only)",
    )
    model_id: str = Field(
        default="mock_model_v1",
        description="Model identifier",
    )
    prediction_time_ms: float = Field(
        ...,
        description="Prediction time in milliseconds",
    )
    is_mock: bool = Field(
        default=True,
        description="Indicates this is a mock response",
    )


class BatchPredictionInput(BaseModel):
    """Input for batch predictions."""

    batch_id: str = Field(..., description="Unique batch identifier")
    features: List[List[float]] = Field(..., description="Feature matrix")


class BatchPredictionOutput(BaseModel):
    """Output for batch predictions."""

    batch_id: str
    total_samples: int
    predictions: List[float]
    processing_time_ms: float
    is_mock: bool = True


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    service: str = "mock_bentoml_service"
    version: str = "1.0.0"
    timestamp: str = ""
    uptime_seconds: float = 0.0


# ============================================================================
# Mock BentoML Service
# ============================================================================

SERVICE_START_TIME = time.time()


@bentoml.service(
    name="mock_model_service",
    resources={"cpu": "1", "memory": "512Mi"},
    traffic={"timeout": 30},
)
class MockModelService:
    """Mock BentoML service for testing infrastructure.

    This service simulates real model predictions with mock data.
    All endpoints mirror the production API.
    """

    def __init__(self):
        """Initialize mock service."""
        self._prediction_count = 0
        self._start_time = SERVICE_START_TIME
        logger.info("Mock BentoML service initialized")

    @bentoml.api
    async def predict(self, input_data: PredictionInput) -> PredictionOutput:
        """Run mock prediction.

        Args:
            input_data: Features and configuration

        Returns:
            Mock predictions
        """
        start = time.time()

        n_samples = len(input_data.features)

        if input_data.model_type == "classification":
            # Binary classification mock
            predictions = [float(random.randint(0, 1)) for _ in range(n_samples)]
            probabilities = [random.uniform(0.3, 0.9) for _ in range(n_samples)]
        else:
            # Regression mock
            predictions = [random.uniform(0, 100) for _ in range(n_samples)]
            probabilities = []

        elapsed_ms = (time.time() - start) * 1000
        self._prediction_count += n_samples

        return PredictionOutput(
            predictions=predictions,
            probabilities=probabilities,
            model_id="mock_model_v1",
            prediction_time_ms=elapsed_ms,
            is_mock=True,
        )

    @bentoml.api
    async def predict_batch(
        self, input_data: BatchPredictionInput
    ) -> BatchPredictionOutput:
        """Run batch mock predictions.

        Args:
            input_data: Batch of features

        Returns:
            Batch predictions
        """
        start = time.time()

        n_samples = len(input_data.features)
        predictions = [random.uniform(0, 1) for _ in range(n_samples)]

        elapsed_ms = (time.time() - start) * 1000
        self._prediction_count += n_samples

        return BatchPredictionOutput(
            batch_id=input_data.batch_id,
            total_samples=n_samples,
            predictions=predictions,
            processing_time_ms=elapsed_ms,
            is_mock=True,
        )

    @bentoml.api
    async def health(self) -> HealthResponse:
        """Health check endpoint.

        Returns:
            Service health status
        """
        uptime = time.time() - self._start_time

        return HealthResponse(
            status="healthy",
            service="mock_bentoml_service",
            version="1.0.0",
            timestamp=datetime.now(timezone.utc).isoformat(),
            uptime_seconds=uptime,
        )

    @bentoml.api
    async def metrics(self) -> Dict[str, Any]:
        """Return service metrics.

        Returns:
            Service metrics for Prometheus
        """
        uptime = time.time() - self._start_time

        return {
            "prediction_count": self._prediction_count,
            "uptime_seconds": uptime,
            "avg_latency_ms": 5.0,  # Mock latency
            "service": "mock_model_service",
            "is_mock": True,
        }

    @bentoml.api
    async def model_info(self) -> Dict[str, Any]:
        """Return model information.

        Returns:
            Model metadata
        """
        return {
            "model_id": "mock_model_v1",
            "model_type": "mock",
            "framework": "mock",
            "version": "1.0.0",
            "is_mock": True,
            "supported_endpoints": [
                "/predict",
                "/predict_batch",
                "/health",
                "/metrics",
                "/model_info",
            ],
        }


# Service instance for BentoML
svc = MockModelService
