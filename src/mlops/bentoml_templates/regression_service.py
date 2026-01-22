"""Regression Service Template for BentoML.

This template provides a production-ready regression service with:
- Continuous value prediction
- Prediction intervals (if supported)
- Batch prediction support
- Health checks and metrics
- Prediction audit trail (Phase 1 G07)

Version: 1.1.0
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Type

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
    BENTOML_AVAILABLE = True
except ImportError:
    BENTOML_AVAILABLE = False
    bentoml = None

try:
    from pydantic import BaseModel, Field
except ImportError:
    BaseModel = object
    Field = lambda *args, **kwargs: None

logger = logging.getLogger(__name__)


# ============================================================================
# Request/Response Models
# ============================================================================


class RegressionInput(BaseModel):
    """Input schema for regression requests."""

    features: List[List[float]] = Field(
        ...,
        description="Feature matrix (samples x features)",
        min_length=1,
    )
    return_intervals: bool = Field(
        default=False,
        description="Return prediction intervals (if supported)",
    )
    confidence_level: float = Field(
        default=0.95,
        ge=0.5,
        le=0.99,
        description="Confidence level for intervals",
    )


class RegressionOutput(BaseModel):
    """Output schema for regression responses."""

    predictions: List[float] = Field(
        ...,
        description="Predicted values",
    )
    lower_bounds: Optional[List[float]] = Field(
        default=None,
        description="Lower bounds of prediction intervals",
    )
    upper_bounds: Optional[List[float]] = Field(
        default=None,
        description="Upper bounds of prediction intervals",
    )
    model_id: str = Field(
        ...,
        description="Model identifier",
    )
    prediction_time_ms: float = Field(
        ...,
        description="Prediction time in milliseconds",
    )


class BatchRegressionInput(BaseModel):
    """Input for batch regression."""

    batch_id: str = Field(
        ...,
        description="Unique batch identifier",
    )
    features: List[List[float]] = Field(
        ...,
        description="Feature matrix",
    )


class BatchRegressionOutput(BaseModel):
    """Output for batch regression."""

    batch_id: str
    total_samples: int
    predictions: List[float]
    mean_prediction: float
    std_prediction: float
    processing_time_ms: float


# ============================================================================
# Regression Service Template
# ============================================================================


class RegressionServiceTemplate:
    """Template for creating regression services.

    This template provides a factory for creating BentoML services
    optimized for regression tasks.

    Example:
        service_class = RegressionServiceTemplate.create(
            model_tag="sales_predictor:latest",
            service_name="sales_prediction_service",
        )
    """

    @staticmethod
    def create(
        model_tag: str,
        service_name: str = "regression_service",
        target_name: str = "target",
        enable_preprocessing: bool = True,
        enable_intervals: bool = False,
        cpu: str = "1",
        memory: str = "2Gi",
        timeout: int = 60,
    ) -> Type:
        """Create a regression service class.

        Args:
            model_tag: BentoML model tag
            service_name: Name for the service
            target_name: Name of the target variable
            enable_preprocessing: Whether to apply preprocessing
            enable_intervals: Whether to compute prediction intervals
            cpu: CPU resource limit
            memory: Memory resource limit
            timeout: Request timeout in seconds

        Returns:
            BentoML service class
        """
        if not BENTOML_AVAILABLE:
            raise ImportError("BentoML is not installed")

        model_ref = bentoml.models.get(model_tag)
        framework = model_ref.info.metadata.get("framework", "sklearn")

        @bentoml.service(
            name=service_name,
            resources={"cpu": cpu, "memory": memory},
            traffic={"timeout": timeout},
        )
        class RegressionService:
            """BentoML regression service."""

            def __init__(self):
                """Initialize service."""
                self.model_tag = model_tag
                self.framework = framework
                self.target_name = target_name
                self._model = None
                self._preprocessor = None
                self._load_model()

                # Metrics tracking
                self._prediction_count = 0
                self._total_latency_ms = 0.0
                self._prediction_sum = 0.0
                self._prediction_sq_sum = 0.0

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
                        custom_objects = getattr(model_ref.info, 'custom_objects', {})
                        if custom_objects:
                            self._preprocessor = custom_objects.get('preprocessor')

                    logger.info(f"Loaded regression model: {self.model_tag}")

                except Exception as e:
                    logger.error(f"Failed to load model: {e}")
                    raise

            def _preprocess(self, features: np.ndarray) -> np.ndarray:
                """Apply preprocessing if available."""
                if self._preprocessor is not None:
                    return self._preprocessor.transform(features)
                return features

            def _compute_intervals(
                self,
                features: np.ndarray,
                predictions: np.ndarray,
                confidence_level: float,
            ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
                """Compute prediction intervals if supported.

                Uses quantile regression or bootstrap if available.
                """
                if not enable_intervals:
                    return None, None

                # Check for quantile prediction support
                if hasattr(self._model, 'predict_quantiles'):
                    alpha = 1 - confidence_level
                    quantiles = self._model.predict_quantiles(
                        features,
                        quantiles=[alpha / 2, 1 - alpha / 2],
                    )
                    return quantiles[:, 0], quantiles[:, 1]

                # Check for RandomForest/GradientBoosting ensemble
                if hasattr(self._model, 'estimators_'):
                    # Bootstrap-like intervals from ensemble
                    individual_preds = np.array([
                        est.predict(features) for est in self._model.estimators_
                    ])
                    alpha = 1 - confidence_level
                    lower = np.percentile(individual_preds, 100 * alpha / 2, axis=0)
                    upper = np.percentile(individual_preds, 100 * (1 - alpha / 2), axis=0)
                    return lower, upper

                # Fallback: estimate from residual std (if available in metadata)
                model_ref = bentoml.models.get(self.model_tag)
                residual_std = model_ref.info.metadata.get('residual_std')
                if residual_std:
                    from scipy import stats
                    z = stats.norm.ppf(1 - (1 - confidence_level) / 2)
                    return predictions - z * residual_std, predictions + z * residual_std

                return None, None

            @bentoml.api
            async def predict(
                self,
                input_data: RegressionInput,
            ) -> RegressionOutput:
                """Run regression prediction.

                Args:
                    input_data: Regression input

                Returns:
                    Regression output with predictions
                """
                start_time = time.time()

                # Convert to numpy
                features = np.array(input_data.features)

                # Preprocess
                features = self._preprocess(features)

                # Get predictions
                predictions = self._model.predict(features)

                # Compute intervals if requested
                lower_bounds, upper_bounds = None, None
                if input_data.return_intervals:
                    lower_bounds, upper_bounds = self._compute_intervals(
                        features,
                        predictions,
                        input_data.confidence_level,
                    )

                # Calculate latency
                elapsed_ms = (time.time() - start_time) * 1000

                # Update metrics
                self._prediction_count += len(predictions)
                self._total_latency_ms += elapsed_ms
                self._prediction_sum += np.sum(predictions)
                self._prediction_sq_sum += np.sum(predictions ** 2)

                output = RegressionOutput(
                    predictions=predictions.tolist(),
                    lower_bounds=lower_bounds.tolist() if lower_bounds is not None else None,
                    upper_bounds=upper_bounds.tolist() if upper_bounds is not None else None,
                    model_id=self.model_tag,
                    prediction_time_ms=elapsed_ms,
                )

                # Prediction audit trail (Phase 1 G07)
                if AUDIT_AVAILABLE and log_prediction_audit:
                    asyncio.create_task(
                        log_prediction_audit(
                            model_name=service_name,
                            model_tag=self.model_tag,
                            service_type="regression",
                            input_data={
                                "n_samples": len(input_data.features),
                                "return_intervals": input_data.return_intervals,
                            },
                            output_data={
                                "predictions_sample": output.predictions[:10] if len(output.predictions) > 10 else output.predictions,
                                "n_predictions": len(output.predictions),
                                "mean_prediction": float(np.mean(predictions)),
                                "has_intervals": output.lower_bounds is not None,
                            },
                            latency_ms=elapsed_ms,
                            metadata={
                                "target_name": self.target_name,
                            },
                        )
                    )

                return output

            @bentoml.api
            async def predict_batch(
                self,
                input_data: BatchRegressionInput,
            ) -> BatchRegressionOutput:
                """Run batch regression.

                Args:
                    input_data: Batch input

                Returns:
                    Batch output with summary statistics
                """
                start_time = time.time()

                features = np.array(input_data.features)
                features = self._preprocess(features)

                predictions = self._model.predict(features)

                elapsed_ms = (time.time() - start_time) * 1000

                output = BatchRegressionOutput(
                    batch_id=input_data.batch_id,
                    total_samples=len(predictions),
                    predictions=predictions.tolist(),
                    mean_prediction=float(np.mean(predictions)),
                    std_prediction=float(np.std(predictions)),
                    processing_time_ms=elapsed_ms,
                )

                # Prediction audit trail (Phase 1 G07)
                if AUDIT_AVAILABLE and log_prediction_audit:
                    asyncio.create_task(
                        log_prediction_audit(
                            model_name=service_name,
                            model_tag=self.model_tag,
                            service_type="regression",
                            input_data={
                                "batch_id": input_data.batch_id,
                                "n_samples": len(input_data.features),
                            },
                            output_data={
                                "batch_id": output.batch_id,
                                "total_samples": output.total_samples,
                                "mean_prediction": output.mean_prediction,
                                "std_prediction": output.std_prediction,
                            },
                            latency_ms=elapsed_ms,
                            metadata={
                                "is_batch": True,
                                "target_name": self.target_name,
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
                    "target_name": self.target_name,
                    "intervals_enabled": enable_intervals,
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
                mean_pred = (
                    self._prediction_sum / self._prediction_count
                    if self._prediction_count > 0
                    else 0.0
                )
                # Variance = E[X^2] - E[X]^2
                var_pred = (
                    (self._prediction_sq_sum / self._prediction_count) - mean_pred ** 2
                    if self._prediction_count > 0
                    else 0.0
                )

                return {
                    "prediction_count": self._prediction_count,
                    "total_latency_ms": self._total_latency_ms,
                    "average_latency_ms": avg_latency,
                    "mean_prediction": mean_pred,
                    "std_prediction": np.sqrt(max(0, var_pred)),
                }

            @bentoml.api
            async def model_info(self) -> Dict[str, Any]:
                """Get model information."""
                model_ref = bentoml.models.get(self.model_tag)
                return {
                    "tag": str(model_ref.tag),
                    "framework": self.framework,
                    "target_name": self.target_name,
                    "intervals_enabled": enable_intervals,
                    "metadata": model_ref.info.metadata,
                    "creation_time": model_ref.info.creation_time.isoformat(),
                }

        return RegressionService
