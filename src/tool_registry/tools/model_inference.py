"""Model Inference Tool for Agent Workflows.

This tool provides agents with the ability to call BentoML model endpoints
for real-time predictions. It integrates with:
- BentoML HTTP endpoints for model serving
- Feast feature store for feature retrieval
- Opik for observability and tracing

Usage:
------
    from src.tool_registry.tools.model_inference import model_inference

    # Direct call
    result = await model_inference(
        model_name="churn_model",
        features={"recency": 10, "frequency": 5, "monetary": 1000.0},
        entity_id="HCP001",
    )

    # Via tool registry
    from src.tool_registry import get_registry
    tool = get_registry().get_callable("model_inference")
    result = await tool(model_name="churn_model", features={...})

Author: E2I Causal Analytics Team
Version: 1.0.0
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union, cast

from pydantic import BaseModel, Field

from src.tool_registry.registry import ToolParameter, ToolSchema, get_registry

logger = logging.getLogger(__name__)


# =============================================================================
# INPUT/OUTPUT SCHEMAS
# =============================================================================


class ModelInferenceInput(BaseModel):
    """Input schema for model inference tool."""

    model_name: str = Field(
        ...,
        description="Name of the model to call (e.g., 'churn_model', 'conversion_model')",
    )
    features: Dict[str, Any] = Field(
        default_factory=dict,
        description="Feature dictionary for prediction. Keys are feature names.",
    )
    entity_id: Optional[str] = Field(
        default=None,
        description="Entity ID for Feast feature retrieval. If provided, features are fetched from store.",
    )
    time_horizon: str = Field(
        default="short_term",
        description="Prediction time horizon: 'short_term', 'medium_term', 'long_term'",
    )
    return_probabilities: bool = Field(
        default=False,
        description="Return class probabilities for classification models",
    )
    return_explanation: bool = Field(
        default=False,
        description="Return SHAP/feature importance explanation",
    )
    trace_context: Optional[Dict[str, str]] = Field(
        default=None,
        description="Opik trace context for distributed tracing",
    )


class ModelInferenceOutput(BaseModel):
    """Output schema for model inference tool."""

    model_name: str = Field(..., description="Name of the model used")
    prediction: Any = Field(..., description="The prediction value")
    confidence: Optional[float] = Field(
        default=None,
        description="Confidence score (0-1) for the prediction",
    )
    probabilities: Optional[Dict[str, float]] = Field(
        default=None,
        description="Class probabilities (classification models only)",
    )
    explanation: Optional[Dict[str, float]] = Field(
        default=None,
        description="Feature importance/SHAP values if requested",
    )
    model_version: Optional[str] = Field(
        default=None,
        description="Version of the model used",
    )
    latency_ms: float = Field(..., description="Inference latency in milliseconds")
    timestamp: str = Field(..., description="Prediction timestamp (ISO format)")
    trace_id: Optional[str] = Field(
        default=None,
        description="Opik trace ID for this inference",
    )
    features_used: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Features that were used (from Feast or input)",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Any warnings from the inference process",
    )


# =============================================================================
# MODEL INFERENCE TOOL
# =============================================================================


@dataclass
class ModelInferenceTool:
    """
    Tool for making model predictions via BentoML endpoints.

    This tool is designed to be used by agents in the E2I platform
    for real-time model inference. It handles:

    1. Feature retrieval from Feast (if entity_id provided)
    2. HTTP call to BentoML endpoint
    3. Response parsing and enrichment
    4. Opik tracing for observability

    Attributes:
        bentoml_base_url: Base URL for BentoML service
        feast_enabled: Whether Feast feature retrieval is enabled
        opik_enabled: Whether Opik tracing is enabled

    Example:
        tool = ModelInferenceTool()
        result = await tool.invoke({
            "model_name": "churn_model",
            "features": {"recency": 10},
        })
    """

    bentoml_base_url: str = field(default="http://localhost:3000")
    feast_enabled: bool = field(default=True)
    opik_enabled: bool = field(default=True)
    _client: Any = field(default=None, repr=False)
    _feast_store: Any = field(default=None, repr=False)

    async def initialize(self) -> None:
        """Initialize HTTP client and feature store connections."""
        if self._client is None:
            try:
                from src.api.dependencies.bentoml_client import get_bentoml_client

                self._client = await get_bentoml_client()
                logger.info("ModelInferenceTool: BentoML client initialized")
            except Exception as e:
                logger.warning(f"BentoML client initialization failed: {e}")

        if self.feast_enabled and self._feast_store is None:
            try:
                from src.feature_store.client import get_feast_store  # type: ignore[attr-defined]

                self._feast_store = await get_feast_store()
                logger.info("ModelInferenceTool: Feast store connected")
            except Exception as e:
                logger.debug(f"Feast store not available: {e}")
                self._feast_store = None

    async def invoke(
        self,
        input_data: Union[Dict[str, Any], ModelInferenceInput],
    ) -> ModelInferenceOutput:
        """
        Make a model prediction.

        Args:
            input_data: Either a dict or ModelInferenceInput with parameters

        Returns:
            ModelInferenceOutput with prediction and metadata
        """
        # Parse input
        if isinstance(input_data, dict):
            params = ModelInferenceInput(**input_data)
        else:
            params = input_data

        # Initialize if needed
        if self._client is None:
            await self.initialize()

        # Start Opik trace if enabled
        trace_id = None
        if self.opik_enabled and params.trace_context:
            trace_id = self._start_trace(params)

        # Get features
        features = params.features
        features_used = dict(features)
        warnings = []

        if params.entity_id and self._feast_store:
            try:
                feast_features = await self._fetch_feast_features(
                    params.entity_id, params.model_name
                )
                # Merge: input features override Feast features
                features_used = {**feast_features, **features}
                features = features_used
            except Exception as e:
                warnings.append(f"Feast feature retrieval failed: {str(e)}")

        # Make prediction
        try:
            result = await self._call_model(
                model_name=params.model_name,
                features=features,
                return_proba=params.return_probabilities,
                return_explanation=params.return_explanation,
                trace_id=trace_id,
            )

            output = ModelInferenceOutput(
                model_name=params.model_name,
                prediction=result.get("prediction"),
                confidence=result.get("confidence"),
                probabilities=result.get("probabilities"),
                explanation=result.get("feature_importance"),
                model_version=result.get("model_version"),
                latency_ms=result.get("_metadata", {}).get("latency_ms", 0),
                timestamp=result.get("_metadata", {}).get(
                    "timestamp", datetime.now(timezone.utc).isoformat()
                ),
                trace_id=trace_id,
                features_used=features_used if params.return_explanation else None,
                warnings=warnings,
            )

        except RuntimeError as e:
            # Circuit breaker open
            output = ModelInferenceOutput(
                model_name=params.model_name,
                prediction=None,
                latency_ms=0,
                timestamp=datetime.now(timezone.utc).isoformat(),
                trace_id=trace_id,
                warnings=[f"Model service unavailable: {str(e)}"],
            )

        except Exception as e:
            logger.error(f"Model inference failed: {e}")
            output = ModelInferenceOutput(
                model_name=params.model_name,
                prediction=None,
                latency_ms=0,
                timestamp=datetime.now(timezone.utc).isoformat(),
                trace_id=trace_id,
                warnings=[f"Inference failed: {str(e)}"],
            )

        # End trace
        if self.opik_enabled and trace_id:
            self._end_trace(trace_id, output)

        return output

    async def _call_model(
        self,
        model_name: str,
        features: Dict[str, Any],
        return_proba: bool = False,
        return_explanation: bool = False,
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Call the BentoML endpoint."""
        if self._client is None:
            raise RuntimeError("BentoML client not initialized")

        input_data = {
            "features": features,
            "return_proba": return_proba,
            "return_explanation": return_explanation,
        }

        result = await self._client.predict(
            model_name=model_name,
            input_data=input_data,
            trace_id=trace_id,
        )

        return cast(Dict[str, Any], result)

    async def _fetch_feast_features(
        self,
        entity_id: str,
        model_name: str,
    ) -> Dict[str, Any]:
        """Fetch features from Feast for an entity."""
        if self._feast_store is None:
            return {}

        # Map model names to feature views
        feature_view_mapping = {
            "churn_model": "hcp_churn_features",
            "conversion_model": "hcp_conversion_features",
            "causal_model": "hcp_causal_features",
        }

        feature_view = feature_view_mapping.get(model_name, f"{model_name}_features")

        try:
            features = await self._feast_store.get_online_features(
                entity_rows=[{"entity_id": entity_id}],
                feature_refs=[f"{feature_view}:*"],
            )
            return features.to_dict() if hasattr(features, "to_dict") else {}
        except Exception as e:
            logger.warning(f"Feast retrieval failed for {entity_id}: {e}")
            return {}

    def _start_trace(self, params: ModelInferenceInput) -> Optional[str]:
        """Start an Opik trace."""
        try:
            import uuid

            import opik

            trace_id = str(uuid.uuid4())
            opik.track(  # type: ignore[call-arg]
                name=f"model_inference.{params.model_name}",
                input={
                    "model_name": params.model_name,
                    "entity_id": params.entity_id,
                    "feature_count": len(params.features),
                },
                metadata={"trace_id": trace_id},
            )
            return trace_id
        except Exception as e:
            logger.debug(f"Opik tracing not available: {e}")
            return None

    def _end_trace(self, trace_id: str, output: ModelInferenceOutput) -> None:
        """End an Opik trace."""
        try:
            import opik

            opik.track(  # type: ignore[call-arg]
                name="model_inference.complete",
                output={
                    "prediction": str(output.prediction)[:100],
                    "confidence": output.confidence,
                    "latency_ms": output.latency_ms,
                },
                metadata={
                    "trace_id": trace_id,
                    "warnings_count": len(output.warnings),
                },
            )
        except Exception:
            pass


# =============================================================================
# SINGLETON AND REGISTRATION
# =============================================================================

_tool_instance: Optional[ModelInferenceTool] = None


async def get_model_inference_tool() -> ModelInferenceTool:
    """Get or create the singleton ModelInferenceTool instance."""
    global _tool_instance
    if _tool_instance is None:
        _tool_instance = ModelInferenceTool()
        await _tool_instance.initialize()
    return _tool_instance


async def model_inference(
    model_name: str,
    features: Optional[Dict[str, Any]] = None,
    entity_id: Optional[str] = None,
    time_horizon: str = "short_term",
    return_probabilities: bool = False,
    return_explanation: bool = False,
    trace_context: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Make a model prediction - the registered tool function.

    This is the function that gets registered in the ToolRegistry
    and can be called by agents.

    Args:
        model_name: Name of the model to call
        features: Feature dictionary (optional if entity_id provided)
        entity_id: Entity ID for Feast feature lookup
        time_horizon: Prediction horizon
        return_probabilities: Include class probabilities
        return_explanation: Include feature importance
        trace_context: Opik trace context

    Returns:
        Dictionary with prediction results
    """
    tool = await get_model_inference_tool()

    result = await tool.invoke(
        ModelInferenceInput(
            model_name=model_name,
            features=features or {},
            entity_id=entity_id,
            time_horizon=time_horizon,
            return_probabilities=return_probabilities,
            return_explanation=return_explanation,
            trace_context=trace_context,
        )
    )

    return result.model_dump()


# =============================================================================
# TOOL REGISTRATION
# =============================================================================


def register_model_inference_tool() -> None:
    """Register the model inference tool in the global registry."""
    schema = ToolSchema(
        name="model_inference",
        description=(
            "Make real-time predictions using BentoML model endpoints. "
            "Supports feature retrieval from Feast and returns predictions "
            "with confidence scores, probabilities, and explanations."
        ),
        source_agent="prediction_synthesizer",
        tier=4,
        input_parameters=[
            ToolParameter(
                name="model_name",
                type="str",
                description="Name of the model (e.g., 'churn_model')",
                required=True,
            ),
            ToolParameter(
                name="features",
                type="Dict[str, Any]",
                description="Feature dictionary for prediction",
                required=False,
                default={},
            ),
            ToolParameter(
                name="entity_id",
                type="str",
                description="Entity ID for Feast feature lookup",
                required=False,
            ),
            ToolParameter(
                name="return_probabilities",
                type="bool",
                description="Return class probabilities",
                required=False,
                default=False,
            ),
            ToolParameter(
                name="return_explanation",
                type="bool",
                description="Return feature importance",
                required=False,
                default=False,
            ),
        ],
        output_schema="ModelInferenceOutput",
        avg_execution_ms=100,
        is_async=True,
        supports_batch=False,
    )

    registry = get_registry()
    registry.register(
        schema=schema,
        callable=model_inference,
        input_model=ModelInferenceInput,
        output_model=ModelInferenceOutput,
    )

    logger.info("Registered model_inference tool in ToolRegistry")


# Auto-register on import (can be disabled if needed)
try:
    register_model_inference_tool()
except Exception as e:
    logger.debug(f"Deferred tool registration: {e}")
