"""Classification Service Template for BentoML.

This template provides a production-ready classification service with:
- Binary and multiclass prediction
- Probability outputs
- Confidence thresholds
- Batch prediction support
- Health checks and metrics
- Prediction audit trail (Phase 1 G07)

Version: 1.1.0
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Type

import numpy as np

# Prediction audit trail (Phase 1 G07)
try:
    from src.mlops.bentoml_prediction_audit import log_prediction_audit

    AUDIT_AVAILABLE = True
except ImportError:
    AUDIT_AVAILABLE = False
    log_prediction_audit = None

try:
    import bentoml

    # Note: bentoml.io module deprecated in v1.4+
    # Use Pydantic models with @bentoml.api decorator instead
    BENTOML_AVAILABLE = True
except ImportError:
    BENTOML_AVAILABLE = False
    bentoml = None

try:
    from pydantic import BaseModel, Field
except ImportError:
    BaseModel = object

    def Field(*args, **kwargs):
        return None


logger = logging.getLogger(__name__)


# ============================================================================
# Request/Response Models
# ============================================================================


class ClassificationInput(BaseModel):
    """Input schema for classification requests."""

    features: List[List[float]] = Field(
        ...,
        description="Feature matrix (samples x features)",
        min_length=1,
    )
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Classification threshold (binary only)",
    )
    return_all_classes: bool = Field(
        default=False,
        description="Return probabilities for all classes",
    )


class ClassificationOutput(BaseModel):
    """Output schema for classification responses."""

    predictions: List[int] = Field(
        ...,
        description="Predicted class labels",
    )
    probabilities: List[float] = Field(
        ...,
        description="Prediction probabilities (positive class for binary)",
    )
    all_probabilities: Optional[List[List[float]]] = Field(
        default=None,
        description="Probabilities for all classes",
    )
    confidence_scores: List[float] = Field(
        ...,
        description="Confidence scores for predictions",
    )
    model_id: str = Field(
        ...,
        description="Model identifier",
    )
    prediction_time_ms: float = Field(
        ...,
        description="Prediction time in milliseconds",
    )


class BatchClassificationInput(BaseModel):
    """Input for batch classification."""

    batch_id: str = Field(
        ...,
        description="Unique batch identifier",
    )
    features: List[List[float]] = Field(
        ...,
        description="Feature matrix",
    )
    threshold: float = Field(default=0.5)


class BatchClassificationOutput(BaseModel):
    """Output for batch classification."""

    batch_id: str
    total_samples: int
    predictions: List[int]
    probabilities: List[float]
    processing_time_ms: float


# ============================================================================
# Classification Service Template
# ============================================================================


class ClassificationServiceTemplate:
    """Template for creating classification services.

    This template provides a factory for creating BentoML services
    optimized for classification tasks.

    Example:
        service_class = ClassificationServiceTemplate.create(
            model_tag="churn_classifier:latest",
            service_name="churn_service",
            n_classes=2,
        )
    """

    @staticmethod
    def create(
        model_tag: str,
        service_name: str = "classification_service",
        n_classes: int = 2,
        class_names: Optional[List[str]] = None,
        default_threshold: float = 0.5,
        enable_preprocessing: bool = True,
        cpu: str = "1",
        memory: str = "2Gi",
        timeout: int = 60,
    ) -> Type:
        """Create a classification service class.

        Args:
            model_tag: BentoML model tag
            service_name: Name for the service
            n_classes: Number of classes
            class_names: Optional class label names
            default_threshold: Default classification threshold
            enable_preprocessing: Whether to apply preprocessing
            cpu: CPU resource limit
            memory: Memory resource limit
            timeout: Request timeout in seconds

        Returns:
            BentoML service class
        """
        if not BENTOML_AVAILABLE:
            raise ImportError("BentoML is not installed")

        # Get model reference for framework detection
        model_ref = bentoml.models.get(model_tag)
        framework = model_ref.info.metadata.get("framework", "sklearn")

        @bentoml.service(
            name=service_name,
            resources={"cpu": cpu, "memory": memory},
            traffic={"timeout": timeout},
        )
        class ClassificationService:
            """BentoML classification service."""

            def __init__(self):
                """Initialize service."""
                self.model_tag = model_tag
                self.framework = framework
                self.n_classes = n_classes
                self.class_names = class_names or [f"class_{i}" for i in range(n_classes)]
                self.default_threshold = default_threshold
                self._model = None
                self._preprocessor = None
                self._load_model()

                # Metrics tracking
                self._prediction_count = 0
                self._total_latency_ms = 0.0

            def _load_model(self):
                """Load model from BentoML store."""
                try:
                    model_ref = bentoml.models.get(self.model_tag)

                    if self.framework == "sklearn":
                        self._model = bentoml.sklearn.load_model(self.model_tag)
                    elif self.framework == "xgboost":
                        self._model = bentoml.xgboost.load_model(self.model_tag)
                    elif self.framework == "lightgbm":
                        self._model = bentoml.lightgbm.load_model(self.model_tag)
                    else:
                        self._model = bentoml.picklable_model.load_model(self.model_tag)

                    # Load preprocessor if available
                    if enable_preprocessing:
                        custom_objects = getattr(model_ref.info, "custom_objects", {})
                        if custom_objects:
                            self._preprocessor = custom_objects.get("preprocessor")

                    logger.info(f"Loaded classification model: {self.model_tag}")

                except Exception as e:
                    logger.error(f"Failed to load model: {e}")
                    raise

            def _preprocess(self, features: np.ndarray) -> np.ndarray:
                """Apply preprocessing if available."""
                if self._preprocessor is not None:
                    return self._preprocessor.transform(features)
                return features

            @bentoml.api
            async def predict(
                self,
                input_data: ClassificationInput,
            ) -> ClassificationOutput:
                """Run classification prediction.

                Args:
                    input_data: Classification input

                Returns:
                    Classification output with predictions and probabilities
                """
                start_time = time.time()

                # Convert to numpy
                features = np.array(input_data.features)

                # Preprocess
                features = self._preprocess(features)

                # Get predictions and probabilities
                predictions = self._model.predict(features)

                # Get probabilities if available
                if hasattr(self._model, "predict_proba"):
                    all_proba = self._model.predict_proba(features)

                    if self.n_classes == 2:
                        # Binary: return positive class probability
                        probabilities = all_proba[:, 1].tolist()
                        # Apply threshold
                        predictions = (
                            (np.array(probabilities) >= input_data.threshold).astype(int).tolist()
                        )
                    else:
                        # Multiclass: return max probability
                        probabilities = np.max(all_proba, axis=1).tolist()
                else:
                    all_proba = None
                    probabilities = [1.0] * len(predictions)

                # Confidence scores (max probability)
                if all_proba is not None:
                    confidence_scores = np.max(all_proba, axis=1).tolist()
                else:
                    confidence_scores = probabilities

                # Calculate latency
                elapsed_ms = (time.time() - start_time) * 1000

                # Update metrics
                self._prediction_count += len(predictions)
                self._total_latency_ms += elapsed_ms

                output = ClassificationOutput(
                    predictions=list(map(int, predictions)),
                    probabilities=probabilities,
                    all_probabilities=all_proba.tolist()
                    if input_data.return_all_classes and all_proba is not None
                    else None,
                    confidence_scores=confidence_scores,
                    model_id=self.model_tag,
                    prediction_time_ms=elapsed_ms,
                )

                # Prediction audit trail (Phase 1 G07)
                if AUDIT_AVAILABLE and log_prediction_audit:
                    asyncio.create_task(
                        log_prediction_audit(
                            model_name=service_name,
                            model_tag=self.model_tag,
                            service_type="classification",
                            input_data={
                                "features": input_data.features,
                                "threshold": input_data.threshold,
                                "n_samples": len(input_data.features),
                            },
                            output_data={
                                "predictions": output.predictions,
                                "probabilities": output.probabilities[:10]
                                if len(output.probabilities) > 10
                                else output.probabilities,
                                "n_predictions": len(output.predictions),
                            },
                            latency_ms=elapsed_ms,
                            metadata={
                                "n_classes": self.n_classes,
                                "class_names": self.class_names,
                            },
                        )
                    )

                return output

            @bentoml.api
            async def predict_batch(
                self,
                input_data: BatchClassificationInput,
            ) -> BatchClassificationOutput:
                """Run batch classification.

                Args:
                    input_data: Batch input

                Returns:
                    Batch output
                """
                start_time = time.time()

                features = np.array(input_data.features)
                features = self._preprocess(features)

                predictions = self._model.predict(features)

                if hasattr(self._model, "predict_proba"):
                    proba = self._model.predict_proba(features)
                    if self.n_classes == 2:
                        probabilities = proba[:, 1].tolist()
                        predictions = (
                            (np.array(probabilities) >= input_data.threshold).astype(int).tolist()
                        )
                    else:
                        probabilities = np.max(proba, axis=1).tolist()
                else:
                    probabilities = [1.0] * len(predictions)

                elapsed_ms = (time.time() - start_time) * 1000

                output = BatchClassificationOutput(
                    batch_id=input_data.batch_id,
                    total_samples=len(predictions),
                    predictions=list(map(int, predictions)),
                    probabilities=probabilities,
                    processing_time_ms=elapsed_ms,
                )

                # Prediction audit trail (Phase 1 G07)
                if AUDIT_AVAILABLE and log_prediction_audit:
                    asyncio.create_task(
                        log_prediction_audit(
                            model_name=service_name,
                            model_tag=self.model_tag,
                            service_type="classification",
                            input_data={
                                "batch_id": input_data.batch_id,
                                "n_samples": len(input_data.features),
                            },
                            output_data={
                                "batch_id": output.batch_id,
                                "total_samples": output.total_samples,
                            },
                            latency_ms=elapsed_ms,
                            metadata={
                                "is_batch": True,
                                "n_classes": self.n_classes,
                            },
                        )
                    )

                return output

            @bentoml.api
            async def health(self) -> Dict[str, Any]:
                """Health check endpoint."""
                return {
                    "status": "healthy",
                    "model_id": self.model_tag,
                    "model_loaded": self._model is not None,
                    "n_classes": self.n_classes,
                    "class_names": self.class_names,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

            @bentoml.api
            async def metrics(self) -> Dict[str, Any]:
                """Get service metrics."""
                avg_latency = (
                    self._total_latency_ms / self._prediction_count
                    if self._prediction_count > 0
                    else 0.0
                )
                return {
                    "prediction_count": self._prediction_count,
                    "total_latency_ms": self._total_latency_ms,
                    "average_latency_ms": avg_latency,
                }

            @bentoml.api
            async def model_info(self) -> Dict[str, Any]:
                """Get model information."""
                model_ref = bentoml.models.get(self.model_tag)
                return {
                    "tag": str(model_ref.tag),
                    "framework": self.framework,
                    "n_classes": self.n_classes,
                    "class_names": self.class_names,
                    "default_threshold": self.default_threshold,
                    "metadata": model_ref.info.metadata,
                    "creation_time": model_ref.info.creation_time.isoformat(),
                }

        return ClassificationService
