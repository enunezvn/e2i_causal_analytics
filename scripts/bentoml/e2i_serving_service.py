"""E2I BentoML Model Serving Service.

Self-contained BentoML service for persistent model serving on the production
droplet. Runs as a standalone process via systemd — NO src.* imports.

Model discovery order:
  1. E2I_BENTOML_MODEL_TAG env var (exact tag, e.g. "tier0_abc123:v5")
  2. E2I_BENTOML_MODEL_NAME env var + ":latest"
  3. Auto-discover latest model from BentoML store
  4. Graceful "no model" mode (health returns degraded, predict returns error)

Framework auto-detection from model metadata:
  - sklearn, xgboost, lightgbm, or pickle fallback

Version: 1.0.0
"""

import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import bentoml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# =============================================================================
# Request/Response Models (matching mock_service.py contract)
# =============================================================================


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
        default="unknown",
        description="Model identifier",
    )
    prediction_time_ms: float = Field(
        ...,
        description="Prediction time in milliseconds",
    )
    is_mock: bool = Field(
        default=False,
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
    is_mock: bool = False


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    service: str = "e2i_model_service"
    version: str = "1.0.0"
    timestamp: str = ""
    uptime_seconds: float = 0.0
    model_loaded: bool = False
    model_tag: Optional[str] = None


# =============================================================================
# Model Discovery
# =============================================================================


def _discover_model() -> tuple[Optional[Any], Optional[str], Optional[str]]:
    """Discover and load a model from the BentoML store.

    Returns:
        (model_object, model_tag_string, framework_name) or (None, None, None)
    """
    import importlib

    # Strategy 1: Exact tag from env
    model_tag = os.environ.get("E2I_BENTOML_MODEL_TAG")
    if model_tag:
        logger.info("Using model tag from E2I_BENTOML_MODEL_TAG: %s", model_tag)
        return _load_model_by_tag(model_tag)

    # Strategy 2: Model name + :latest from env
    model_name = os.environ.get("E2I_BENTOML_MODEL_NAME")
    if model_name:
        tag = f"{model_name}:latest"
        logger.info("Using model name from E2I_BENTOML_MODEL_NAME: %s", tag)
        return _load_model_by_tag(tag)

    # Strategy 3: Auto-discover latest model from store
    try:
        models = bentoml.models.list()
        if models:
            # Sort by creation time (newest first) — models.list() returns
            # BentoModel objects with a creation_time attribute
            sorted_models = sorted(
                models,
                key=lambda m: getattr(m, "creation_time", ""),
                reverse=True,
            )
            latest = sorted_models[0]
            tag_str = str(latest.tag)
            logger.info("Auto-discovered latest model: %s", tag_str)
            return _load_model_by_tag(tag_str)
        else:
            logger.warning("No models found in BentoML store")
    except Exception as e:
        logger.warning("Failed to list models from BentoML store: %s", e)

    # Strategy 4: No model — graceful degraded mode
    logger.warning("No model available — running in degraded mode")
    return None, None, None


def _load_model_by_tag(tag: str) -> tuple[Optional[Any], Optional[str], Optional[str]]:
    """Load a model by its BentoML tag string.

    Tries framework-specific loaders, falls back to pickle.

    Returns:
        (model_object, tag_string, framework_name)
    """
    framework_loaders = [
        ("sklearn", "bentoml.sklearn"),
        ("xgboost", "bentoml.xgboost"),
        ("lightgbm", "bentoml.lightgbm"),
    ]

    # Try to read model metadata to detect framework
    try:
        bento_model = bentoml.models.get(tag)
        meta = bento_model.info.metadata or {}
        stored_framework = meta.get("framework", "").lower()
        if stored_framework:
            # Reorder loaders to try the stored framework first
            framework_loaders = sorted(
                framework_loaders,
                key=lambda fl: (0 if fl[0] == stored_framework else 1),
            )
    except Exception:
        pass

    # Try each framework loader
    for framework_name, module_path in framework_loaders:
        try:
            mod = __import__(module_path, fromlist=[module_path.split(".")[-1]])
            model = mod.load_model(tag)
            logger.info("Loaded model %s with %s loader", tag, framework_name)
            return model, tag, framework_name
        except (bentoml.exceptions.NotFound, Exception):
            continue

    # Pickle fallback
    try:
        import pickle

        bento_model = bentoml.models.get(tag)
        model_path = bento_model.path
        # Look for common pickle filenames
        for pkl_name in ["saved_model.pkl", "model.pkl", "model.joblib"]:
            pkl_path = os.path.join(model_path, pkl_name)
            if os.path.exists(pkl_path):
                with open(pkl_path, "rb") as f:
                    model = pickle.load(f)  # noqa: S301
                logger.info("Loaded model %s via pickle (%s)", tag, pkl_name)
                return model, tag, "pickle"
    except Exception as e:
        logger.error("Pickle fallback failed for %s: %s", tag, e)

    logger.error("Failed to load model: %s", tag)
    return None, None, None


# =============================================================================
# BentoML Service
# =============================================================================

SERVICE_START_TIME = time.time()


@bentoml.service(
    name="e2i_model_service",
    resources={"cpu": "1", "memory": "2Gi"},
    traffic={"timeout": 60},
)
class E2IModelService:
    """Production BentoML service for E2I trained models.

    Provides persistent model serving with automatic model discovery
    and framework detection. Matches the mock_service.py API contract.
    """

    def __init__(self):
        """Initialize service and load model."""
        self._start_time = SERVICE_START_TIME
        self._prediction_count = 0
        self._model = None
        self._model_tag: Optional[str] = None
        self._framework: Optional[str] = None

        self._model, self._model_tag, self._framework = _discover_model()

        if self._model is not None:
            logger.info(
                "E2I Model Service initialized: tag=%s framework=%s",
                self._model_tag,
                self._framework,
            )
        else:
            logger.warning("E2I Model Service initialized in degraded mode (no model)")

    def _run_prediction(self, features: List[List[float]]) -> PredictionOutput:
        """Run prediction using the loaded model.

        Args:
            features: Feature matrix

        Returns:
            Prediction output
        """
        import numpy as np

        if self._model is None:
            return PredictionOutput(
                predictions=[],
                probabilities=[],
                model_id="no_model",
                prediction_time_ms=0.0,
                is_mock=False,
            )

        start = time.time()
        arr = np.array(features)

        predictions = self._model.predict(arr).tolist()

        probabilities = []
        if hasattr(self._model, "predict_proba"):
            try:
                proba = self._model.predict_proba(arr)
                # Return probability of positive class for binary classification
                if proba.ndim == 2 and proba.shape[1] == 2:
                    probabilities = proba[:, 1].tolist()
                else:
                    probabilities = proba.tolist()
            except Exception:
                pass

        elapsed_ms = (time.time() - start) * 1000
        self._prediction_count += len(features)

        return PredictionOutput(
            predictions=predictions,
            probabilities=probabilities,
            model_id=self._model_tag or "unknown",
            prediction_time_ms=elapsed_ms,
            is_mock=False,
        )

    @bentoml.api
    async def predict(self, input_data: PredictionInput) -> PredictionOutput:
        """Run prediction on input features.

        Args:
            input_data: Features and configuration

        Returns:
            Model predictions
        """
        return self._run_prediction(input_data.features)

    @bentoml.api
    async def predict_batch(
        self, input_data: BatchPredictionInput
    ) -> BatchPredictionOutput:
        """Run batch predictions.

        Args:
            input_data: Batch of features

        Returns:
            Batch predictions
        """
        start = time.time()

        if self._model is None:
            return BatchPredictionOutput(
                batch_id=input_data.batch_id,
                total_samples=len(input_data.features),
                predictions=[],
                processing_time_ms=0.0,
            )

        import numpy as np

        arr = np.array(input_data.features)
        predictions = self._model.predict(arr).tolist()
        elapsed_ms = (time.time() - start) * 1000
        self._prediction_count += len(input_data.features)

        return BatchPredictionOutput(
            batch_id=input_data.batch_id,
            total_samples=len(input_data.features),
            predictions=predictions,
            processing_time_ms=elapsed_ms,
        )

    @bentoml.api
    async def health(self) -> HealthResponse:
        """Health check endpoint.

        Returns:
            Service health status
        """
        uptime = time.time() - self._start_time
        status = "healthy" if self._model is not None else "degraded"

        return HealthResponse(
            status=status,
            service="e2i_model_service",
            version="1.0.0",
            timestamp=datetime.now(timezone.utc).isoformat(),
            uptime_seconds=uptime,
            model_loaded=self._model is not None,
            model_tag=self._model_tag,
        )

    @bentoml.api
    async def metrics(self) -> Dict[str, Any]:
        """Return service metrics.

        Returns:
            Service metrics
        """
        uptime = time.time() - self._start_time

        return {
            "prediction_count": self._prediction_count,
            "uptime_seconds": uptime,
            "model_tag": self._model_tag,
            "framework": self._framework,
            "model_loaded": self._model is not None,
            "service": "e2i_model_service",
            "is_mock": False,
        }

    @bentoml.api
    async def model_info(self) -> Dict[str, Any]:
        """Return model information.

        Returns:
            Model metadata
        """
        info: Dict[str, Any] = {
            "model_id": self._model_tag or "no_model",
            "model_type": self._framework or "none",
            "framework": self._framework or "none",
            "version": "1.0.0",
            "is_mock": False,
            "model_loaded": self._model is not None,
            "supported_endpoints": [
                "/predict",
                "/predict_batch",
                "/health",
                "/metrics",
                "/model_info",
            ],
        }

        # Add model metadata if available
        if self._model_tag:
            try:
                bento_model = bentoml.models.get(self._model_tag)
                meta = bento_model.info.metadata or {}
                info["metadata"] = meta
            except Exception:
                pass

        return info
