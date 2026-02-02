"""Causal Inference Service Template for BentoML.

This template provides a production-ready causal inference service with:
- CATE (Conditional Average Treatment Effect) estimation
- Treatment effect predictions
- Confidence intervals
- Batch processing support
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


class CausalInput(BaseModel):
    """Input schema for causal inference requests."""

    features: List[List[float]] = Field(
        ...,
        description="Feature matrix (samples x features)",
        min_length=1,
    )
    treatment: Optional[List[int]] = Field(
        default=None,
        description="Treatment assignment (0 or 1) for each sample",
    )
    return_intervals: bool = Field(
        default=True,
        description="Return confidence intervals for CATE",
    )
    confidence_level: float = Field(
        default=0.95,
        ge=0.5,
        le=0.99,
        description="Confidence level for intervals",
    )


class CATEOutput(BaseModel):
    """Output schema for CATE estimation."""

    cate: List[float] = Field(
        ...,
        description="Conditional Average Treatment Effect estimates",
    )
    lower_bounds: Optional[List[float]] = Field(
        default=None,
        description="Lower bounds of confidence intervals",
    )
    upper_bounds: Optional[List[float]] = Field(
        default=None,
        description="Upper bounds of confidence intervals",
    )
    ate: float = Field(
        ...,
        description="Average Treatment Effect (mean of CATE)",
    )
    ate_std: float = Field(
        ...,
        description="Standard deviation of ATE",
    )
    model_id: str = Field(
        ...,
        description="Model identifier",
    )
    prediction_time_ms: float = Field(
        ...,
        description="Prediction time in milliseconds",
    )


class TreatmentEffectInput(BaseModel):
    """Input for treatment effect prediction."""

    features: List[List[float]] = Field(
        ...,
        description="Feature matrix",
    )
    treatment_values: List[int] = Field(
        default=[0, 1],
        description="Treatment values to compare",
    )


class TreatmentEffectOutput(BaseModel):
    """Output for treatment effect prediction."""

    potential_outcomes: Dict[str, List[float]] = Field(
        ...,
        description="Potential outcomes for each treatment value",
    )
    treatment_effects: List[float] = Field(
        ...,
        description="Individual treatment effects (Y1 - Y0)",
    )
    mean_effect: float = Field(
        ...,
        description="Mean treatment effect",
    )
    effect_std: float = Field(
        ...,
        description="Standard deviation of effects",
    )


class BatchCausalInput(BaseModel):
    """Input for batch causal inference."""

    batch_id: str = Field(
        ...,
        description="Unique batch identifier",
    )
    features: List[List[float]] = Field(
        ...,
        description="Feature matrix",
    )


class BatchCausalOutput(BaseModel):
    """Output for batch causal inference."""

    batch_id: str
    total_samples: int
    cate: List[float]
    ate: float
    ate_std: float
    positive_effect_ratio: float
    processing_time_ms: float


# ============================================================================
# Causal Inference Service Template
# ============================================================================


class CausalInferenceServiceTemplate:
    """Template for creating causal inference services.

    This template provides a factory for creating BentoML services
    optimized for causal inference tasks using EconML/DoWhy models.

    Example:
        service_class = CausalInferenceServiceTemplate.create(
            model_tag="cate_model:latest",
            service_name="cate_service",
        )
    """

    @staticmethod
    def create(
        model_tag: str,
        service_name: str = "causal_inference_service",
        treatment_name: str = "treatment",
        outcome_name: str = "outcome",
        enable_preprocessing: bool = True,
        cpu: str = "2",
        memory: str = "4Gi",
        timeout: int = 120,
    ) -> Type:
        """Create a causal inference service class.

        Args:
            model_tag: BentoML model tag
            service_name: Name for the service
            treatment_name: Name of the treatment variable
            outcome_name: Name of the outcome variable
            enable_preprocessing: Whether to apply preprocessing
            cpu: CPU resource limit
            memory: Memory resource limit
            timeout: Request timeout in seconds

        Returns:
            BentoML service class
        """
        if not BENTOML_AVAILABLE:
            raise ImportError("BentoML is not installed")

        model_ref = bentoml.models.get(model_tag)
        framework = model_ref.info.metadata.get("framework", "econml")

        @bentoml.service(
            name=service_name,
            resources={"cpu": cpu, "memory": memory},
            traffic={"timeout": timeout},
        )
        class CausalInferenceService:
            """BentoML causal inference service."""

            def __init__(self):
                """Initialize service."""
                self.model_tag = model_tag
                self.framework = framework
                self.treatment_name = treatment_name
                self.outcome_name = outcome_name
                self._model = None
                self._preprocessor = None
                self._load_model()

                # Metrics tracking
                self._prediction_count = 0
                self._total_latency_ms = 0.0
                self._cate_sum = 0.0
                self._cate_sq_sum = 0.0

            def _load_model(self):
                """Load model from BentoML store."""
                try:
                    model_ref = bentoml.models.get(self.model_tag)

                    # EconML models are typically pickled
                    self._model = bentoml.picklable_model.load_model(self.model_tag)

                    # Load preprocessor if available
                    if enable_preprocessing:
                        custom_objects = getattr(model_ref.info, "custom_objects", {})
                        if custom_objects:
                            self._preprocessor = custom_objects.get("preprocessor")

                    logger.info(f"Loaded causal model: {self.model_tag}")

                except Exception as e:
                    logger.error(f"Failed to load model: {e}")
                    raise

            def _preprocess(self, features: np.ndarray) -> np.ndarray:
                """Apply preprocessing if available."""
                if self._preprocessor is not None:
                    return self._preprocessor.transform(features)
                return features

            def _estimate_cate(
                self,
                features: np.ndarray,
            ) -> np.ndarray:
                """Estimate CATE using the loaded model.

                Supports EconML estimators (DML, CausalForest, etc.)
                """
                # EconML models use effect() or const_marginal_effect()
                if hasattr(self._model, "effect"):
                    return self._model.effect(features)
                elif hasattr(self._model, "const_marginal_effect"):
                    return self._model.const_marginal_effect(features)
                elif hasattr(self._model, "predict"):
                    return self._model.predict(features)
                else:
                    raise ValueError("Model does not support CATE estimation")

            def _estimate_intervals(
                self,
                features: np.ndarray,
                confidence_level: float,
            ) -> tuple:
                """Estimate confidence intervals for CATE."""
                alpha = 1 - confidence_level

                # EconML models with inference support
                if hasattr(self._model, "effect_interval"):
                    lower, upper = self._model.effect_interval(
                        features,
                        alpha=alpha,
                    )
                    return lower.flatten(), upper.flatten()

                if hasattr(self._model, "const_marginal_effect_interval"):
                    lower, upper = self._model.const_marginal_effect_interval(
                        features,
                        alpha=alpha,
                    )
                    return lower.flatten(), upper.flatten()

                # Fallback: bootstrap if model has fit method
                if hasattr(self._model, "effect_inference"):
                    inference = self._model.effect_inference(features)
                    lower, upper = inference.conf_int(alpha=alpha)
                    return lower.flatten(), upper.flatten()

                return None, None

            @bentoml.api
            async def estimate_cate(
                self,
                input_data: CausalInput,
            ) -> CATEOutput:
                """Estimate Conditional Average Treatment Effect.

                Args:
                    input_data: Causal input with features

                Returns:
                    CATE estimates with optional confidence intervals
                """
                start_time = time.time()

                # Convert to numpy
                features = np.array(input_data.features)

                # Preprocess
                features = self._preprocess(features)

                # Get CATE estimates
                cate = self._estimate_cate(features)
                if cate.ndim > 1:
                    cate = cate.flatten()

                # Compute confidence intervals if requested
                lower_bounds, upper_bounds = None, None
                if input_data.return_intervals:
                    lower_bounds, upper_bounds = self._estimate_intervals(
                        features,
                        input_data.confidence_level,
                    )

                # Calculate ATE statistics
                ate = float(np.mean(cate))
                ate_std = float(np.std(cate))

                # Calculate latency
                elapsed_ms = (time.time() - start_time) * 1000

                # Update metrics
                self._prediction_count += len(cate)
                self._total_latency_ms += elapsed_ms
                self._cate_sum += np.sum(cate)
                self._cate_sq_sum += np.sum(cate**2)

                output = CATEOutput(
                    cate=cate.tolist(),
                    lower_bounds=lower_bounds.tolist() if lower_bounds is not None else None,
                    upper_bounds=upper_bounds.tolist() if upper_bounds is not None else None,
                    ate=ate,
                    ate_std=ate_std,
                    model_id=self.model_tag,
                    prediction_time_ms=elapsed_ms,
                )

                # Prediction audit trail (Phase 1 G07)
                if AUDIT_AVAILABLE and log_prediction_audit:
                    asyncio.create_task(
                        log_prediction_audit(
                            model_name=service_name,
                            model_tag=self.model_tag,
                            service_type="causal",
                            input_data={
                                "n_samples": len(input_data.features),
                                "return_intervals": input_data.return_intervals,
                                "confidence_level": input_data.confidence_level,
                            },
                            output_data={
                                "ate": output.ate,
                                "ate_std": output.ate_std,
                                "n_cate_estimates": len(output.cate),
                                "has_intervals": output.lower_bounds is not None,
                                "cate_sample": output.cate[:10]
                                if len(output.cate) > 10
                                else output.cate,
                            },
                            latency_ms=elapsed_ms,
                            metadata={
                                "treatment_name": self.treatment_name,
                                "outcome_name": self.outcome_name,
                                "framework": self.framework,
                            },
                        )
                    )

                return output

            @bentoml.api
            async def estimate_treatment_effects(
                self,
                input_data: TreatmentEffectInput,
            ) -> TreatmentEffectOutput:
                """Estimate potential outcomes and treatment effects.

                Args:
                    input_data: Treatment effect input

                Returns:
                    Potential outcomes and individual treatment effects
                """
                features = np.array(input_data.features)
                features = self._preprocess(features)

                treatment_values = input_data.treatment_values
                potential_outcomes = {}

                # Estimate potential outcomes for each treatment value
                for t in treatment_values:
                    if hasattr(self._model, "effect"):
                        # For DML-style models, effect is already the difference
                        if t == 1:
                            potential_outcomes[f"T={t}"] = (
                                self._model.effect(features).flatten().tolist()
                            )
                        else:
                            # T=0 is baseline
                            potential_outcomes[f"T={t}"] = [0.0] * len(features)
                    elif hasattr(self._model, "predict"):
                        # For models that predict outcomes directly
                        treatment_array = np.full(len(features), t)
                        outcomes = self._model.predict(features, treatment_array)
                        potential_outcomes[f"T={t}"] = outcomes.tolist()

                # Calculate treatment effects (Y1 - Y0)
                if "T=1" in potential_outcomes and "T=0" in potential_outcomes:
                    effects = np.array(potential_outcomes["T=1"]) - np.array(
                        potential_outcomes["T=0"]
                    )
                else:
                    # If only effect available
                    effects = np.array(potential_outcomes.get("T=1", [0.0] * len(features)))

                output = TreatmentEffectOutput(
                    potential_outcomes=potential_outcomes,
                    treatment_effects=effects.tolist(),
                    mean_effect=float(np.mean(effects)),
                    effect_std=float(np.std(effects)),
                )

                # Prediction audit trail (Phase 1 G07)
                if AUDIT_AVAILABLE and log_prediction_audit:
                    asyncio.create_task(
                        log_prediction_audit(
                            model_name=service_name,
                            model_tag=self.model_tag,
                            service_type="causal",
                            input_data={
                                "n_samples": len(input_data.features),
                                "treatment_values": input_data.treatment_values,
                            },
                            output_data={
                                "mean_effect": output.mean_effect,
                                "effect_std": output.effect_std,
                                "n_effects": len(output.treatment_effects),
                            },
                            latency_ms=0.0,  # Not timed separately
                            metadata={
                                "method": "estimate_treatment_effects",
                                "treatment_name": self.treatment_name,
                            },
                        )
                    )

                return output

            @bentoml.api
            async def estimate_batch(
                self,
                input_data: BatchCausalInput,
            ) -> BatchCausalOutput:
                """Run batch CATE estimation.

                Args:
                    input_data: Batch input

                Returns:
                    Batch output with summary statistics
                """
                start_time = time.time()

                features = np.array(input_data.features)
                features = self._preprocess(features)

                cate = self._estimate_cate(features)
                if cate.ndim > 1:
                    cate = cate.flatten()

                elapsed_ms = (time.time() - start_time) * 1000

                # Calculate proportion with positive treatment effect
                positive_effect_ratio = float(np.mean(cate > 0))

                output = BatchCausalOutput(
                    batch_id=input_data.batch_id,
                    total_samples=len(cate),
                    cate=cate.tolist(),
                    ate=float(np.mean(cate)),
                    ate_std=float(np.std(cate)),
                    positive_effect_ratio=positive_effect_ratio,
                    processing_time_ms=elapsed_ms,
                )

                # Prediction audit trail (Phase 1 G07)
                if AUDIT_AVAILABLE and log_prediction_audit:
                    asyncio.create_task(
                        log_prediction_audit(
                            model_name=service_name,
                            model_tag=self.model_tag,
                            service_type="causal",
                            input_data={
                                "batch_id": input_data.batch_id,
                                "n_samples": len(input_data.features),
                            },
                            output_data={
                                "batch_id": output.batch_id,
                                "total_samples": output.total_samples,
                                "ate": output.ate,
                                "ate_std": output.ate_std,
                                "positive_effect_ratio": output.positive_effect_ratio,
                            },
                            latency_ms=elapsed_ms,
                            metadata={
                                "is_batch": True,
                                "treatment_name": self.treatment_name,
                                "outcome_name": self.outcome_name,
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
                    "treatment_name": self.treatment_name,
                    "outcome_name": self.outcome_name,
                    "framework": self.framework,
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
                mean_cate = (
                    self._cate_sum / self._prediction_count if self._prediction_count > 0 else 0.0
                )
                # Variance = E[X^2] - E[X]^2
                var_cate = (
                    (self._cate_sq_sum / self._prediction_count) - mean_cate**2
                    if self._prediction_count > 0
                    else 0.0
                )

                return {
                    "prediction_count": self._prediction_count,
                    "total_latency_ms": self._total_latency_ms,
                    "average_latency_ms": avg_latency,
                    "mean_cate": mean_cate,
                    "std_cate": np.sqrt(max(0, var_cate)),
                }

            @bentoml.api
            async def model_info(self) -> Dict[str, Any]:
                """Get model information."""
                model_ref = bentoml.models.get(self.model_tag)
                return {
                    "tag": str(model_ref.tag),
                    "framework": self.framework,
                    "treatment_name": self.treatment_name,
                    "outcome_name": self.outcome_name,
                    "metadata": model_ref.info.metadata,
                    "creation_time": model_ref.info.creation_time.isoformat(),
                }

        return CausalInferenceService
