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
Version: 4.1.0
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from enum import Enum
import uuid
import logging

# Internal imports (adjust paths for your project structure)
# from src.mlops.bentoml_service import BentoMLClient
# from src.mlops.shap_explainer import RealTimeSHAPExplainer
# from src.database.repositories.ml_shap_analysis import MLSHAPAnalysisRepository
# from src.database.repositories.ml_prediction import MLPredictionRepository

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/explain", tags=["Model Interpretability"])


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
    FULL = "full"              # All SHAP values + metadata
    TOP_K = "top_k"            # Only top K contributing features
    NARRATIVE = "narrative"    # NL explanation (requires Claude)
    MINIMAL = "minimal"        # Prediction + top 3 features only


class FeatureContribution(BaseModel):
    """Single feature's contribution to prediction."""
    feature_name: str = Field(..., description="Name of the feature")
    feature_value: Any = Field(..., description="Actual value of feature for this instance")
    shap_value: float = Field(..., description="SHAP contribution to prediction")
    contribution_direction: str = Field(..., description="positive or negative")
    contribution_rank: int = Field(..., description="Rank by absolute SHAP value")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "feature_name": "days_since_last_hcp_visit",
            "feature_value": 45,
            "shap_value": 0.234,
            "contribution_direction": "positive",
            "contribution_rank": 1
        }
    })


class ExplainRequest(BaseModel):
    """Request payload for real-time explanation."""
    patient_id: str = Field(..., description="Patient identifier")
    hcp_id: Optional[str] = Field(None, description="HCP context for the prediction")
    model_type: ModelType = Field(..., description="Type of model to explain")
    model_version_id: Optional[str] = Field(None, description="Specific model version (latest if not specified)")
    features: Optional[Dict[str, Any]] = Field(None, description="Pre-computed features (fetched from Feast if not provided)")
    format: ExplanationFormat = Field(default=ExplanationFormat.TOP_K, description="Output format")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of top features to return")
    include_base_value: bool = Field(default=True, description="Include model's base prediction value")
    store_for_audit: bool = Field(default=True, description="Store explanation in ml_shap_analyses for compliance")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "patient_id": "PAT-2024-001234",
            "hcp_id": "HCP-NE-5678",
            "model_type": "propensity",
            "format": "top_k",
            "top_k": 5,
            "store_for_audit": True
        }
    })


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
    base_value: Optional[float] = Field(None, description="Model's expected value (average prediction)")
    top_features: List[FeatureContribution] = Field(..., description="Top contributing features")
    shap_sum: float = Field(..., description="Sum of all SHAP values (should equal prediction - base_value)")
    
    # Optional narrative
    narrative_explanation: Optional[str] = Field(None, description="Natural language explanation (if format=narrative)")
    
    # Metadata
    computation_time_ms: float = Field(..., description="Time to compute explanation")
    audit_stored: bool = Field(..., description="Whether explanation was stored for compliance")

    model_config = ConfigDict(json_schema_extra={
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
                    "contribution_rank": 1
                }
            ],
            "shap_sum": 0.36,
            "narrative_explanation": None,
            "computation_time_ms": 127.5,
            "audit_stored": True
        }
    })


class BatchExplainRequest(BaseModel):
    """Batch explanation request for multiple patients."""
    requests: List[ExplainRequest] = Field(..., max_length=50, description="Up to 50 patients per batch")
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
    
    def __init__(self):
        # These would be injected in production
        self.bentoml_client = None  # BentoMLClient()
        self.shap_explainer = None  # RealTimeSHAPExplainer()
        self.shap_repo = None       # MLSHAPAnalysisRepository()
        self.feast_client = None    # FeastClient()
        
    async def get_features(self, patient_id: str, model_type: ModelType) -> Dict[str, Any]:
        """Retrieve features from Feast feature store."""
        # In production: return await self.feast_client.get_online_features(patient_id, model_type)
        # Mock for demonstration
        return {
            "days_since_last_hcp_visit": 45,
            "total_hcp_interactions_90d": 12,
            "therapy_adherence_score": 0.72,
            "lab_value_trend": "improving",
            "prior_brand_experience": True,
            "insurance_tier": "commercial",
            "region": "northeast",
            "hcp_specialty_match": True,
            "patient_age_bucket": "45-54",
            "comorbidity_count": 2
        }
    
    async def get_prediction(
        self, 
        features: Dict[str, Any], 
        model_type: ModelType,
        model_version_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get prediction from BentoML endpoint."""
        # In production: return await self.bentoml_client.predict(features, model_type, model_version_id)
        # Mock for demonstration
        return {
            "prediction_class": "high_propensity",
            "prediction_probability": 0.78,
            "model_version_id": model_version_id or "v2.3.1-prod"
        }
    
    async def compute_shap(
        self,
        features: Dict[str, Any],
        model_type: ModelType,
        model_version_id: str,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Compute SHAP values for a single instance.
        
        Uses TreeExplainer for tree-based models (fast),
        KernelExplainer for others (slower).
        """
        # In production:
        # explainer = self.shap_explainer.get_explainer(model_type, model_version_id)
        # shap_values = explainer.shap_values(features)
        # base_value = explainer.expected_value
        
        # Mock SHAP computation
        import random
        feature_names = list(features.keys())
        shap_values = {name: round(random.uniform(-0.2, 0.3), 4) for name in feature_names}
        base_value = 0.42
        
        # Sort by absolute value and take top K
        sorted_features = sorted(
            shap_values.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )[:top_k]
        
        contributions = []
        for rank, (name, shap_val) in enumerate(sorted_features, 1):
            contributions.append(FeatureContribution(
                feature_name=name,
                feature_value=features[name],
                shap_value=shap_val,
                contribution_direction="positive" if shap_val > 0 else "negative",
                contribution_rank=rank
            ))
        
        return {
            "base_value": base_value,
            "contributions": contributions,
            "shap_sum": sum(shap_values.values())
        }
    
    async def generate_narrative(
        self,
        patient_id: str,
        prediction: Dict[str, Any],
        contributions: List[FeatureContribution]
    ) -> str:
        """
        Generate natural language explanation using Claude.
        
        This would call the explainer agent for narrative generation.
        """
        # In production: call explainer agent
        top_factors = ", ".join([c.feature_name for c in contributions[:3]])
        return (
            f"This patient shows {prediction['prediction_class']} based primarily on: {top_factors}. "
            f"The model confidence is {prediction['prediction_probability']:.0%}."
        )
    
    async def store_audit_record(
        self,
        explanation_id: str,
        patient_id: str,
        model_type: str,
        model_version_id: str,
        features: Dict[str, Any],
        shap_values: Dict[str, float],
        prediction: Dict[str, Any]
    ) -> bool:
        """Store explanation in ml_shap_analyses for regulatory audit."""
        # In production:
        # await self.shap_repo.create({
        #     "shap_analysis_id": explanation_id,
        #     "model_version_id": model_version_id,
        #     "analysis_type": "local_realtime",
        #     "input_features": features,
        #     "shap_values": shap_values,
        #     "prediction_context": {"patient_id": patient_id, **prediction},
        #     "created_at": datetime.now(timezone.utc)
        # })
        logger.info(f"Stored audit record for explanation {explanation_id}")
        return True


def get_shap_service() -> RealTimeSHAPService:
    """Dependency injection for SHAP service."""
    return RealTimeSHAPService()


# =============================================================================
# ENDPOINTS
# =============================================================================

@router.post(
    "/predict",
    response_model=ExplainResponse,
    summary="Get prediction with real-time SHAP explanation",
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
    """
)
async def explain_prediction(
    request: ExplainRequest,
    background_tasks: BackgroundTasks,
    service: RealTimeSHAPService = Depends(get_shap_service)
) -> ExplainResponse:
    """
    Real-time prediction with SHAP explanation.
    """
    import time
    start_time = time.time()
    
    explanation_id = f"EXPL-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{uuid.uuid4().hex[:8]}"
    
    try:
        # 1. Get features (from request or Feast)
        features = request.features
        if features is None:
            features = await service.get_features(request.patient_id, request.model_type)
        
        # 2. Get prediction from BentoML
        prediction = await service.get_prediction(
            features=features,
            model_type=request.model_type,
            model_version_id=request.model_version_id
        )
        
        # 3. Compute SHAP values
        shap_result = await service.compute_shap(
            features=features,
            model_type=request.model_type,
            model_version_id=prediction["model_version_id"],
            top_k=request.top_k
        )
        
        # 4. Generate narrative (if requested)
        narrative = None
        if request.format == ExplanationFormat.NARRATIVE:
            narrative = await service.generate_narrative(
                patient_id=request.patient_id,
                prediction=prediction,
                contributions=shap_result["contributions"]
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
                shap_values={c.feature_name: c.shap_value for c in shap_result["contributions"]},
                prediction=prediction
            )
            audit_stored = True
        
        computation_time_ms = (time.time() - start_time) * 1000
        
        return ExplainResponse(
            explanation_id=explanation_id,
            request_timestamp=datetime.now(timezone.utc),
            patient_id=request.patient_id,
            model_type=request.model_type,
            model_version_id=prediction["model_version_id"],
            prediction_class=prediction["prediction_class"],
            prediction_probability=prediction["prediction_probability"],
            base_value=shap_result["base_value"] if request.include_base_value else None,
            top_features=shap_result["contributions"],
            shap_sum=shap_result["shap_sum"],
            narrative_explanation=narrative,
            computation_time_ms=round(computation_time_ms, 2),
            audit_stored=audit_stored
        )
        
    except Exception as e:
        logger.error(f"Explanation failed for patient {request.patient_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")


@router.post(
    "/predict/batch",
    response_model=BatchExplainResponse,
    summary="Batch predictions with SHAP explanations",
    description="Process up to 50 patients in a single request. Useful for pre-computing explanations."
)
async def explain_batch(
    request: BatchExplainRequest,
    background_tasks: BackgroundTasks,
    service: RealTimeSHAPService = Depends(get_shap_service)
) -> BatchExplainResponse:
    """
    Batch explanation endpoint for multiple patients.
    """
    import time
    import asyncio
    
    start_time = time.time()
    batch_id = f"BATCH-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{uuid.uuid4().hex[:8]}"
    
    explanations = []
    errors = []
    
    async def process_single(req: ExplainRequest) -> Optional[ExplainResponse]:
        try:
            return await explain_prediction(req, background_tasks, service)
        except HTTPException as e:
            errors.append({"patient_id": req.patient_id, "error": e.detail})
            return None
    
    if request.parallel:
        results = await asyncio.gather(
            *[process_single(req) for req in request.requests],
            return_exceptions=True
        )
        explanations = [r for r in results if isinstance(r, ExplainResponse)]
    else:
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
        total_time_ms=round(total_time_ms, 2)
    )


@router.get(
    "/history/{patient_id}",
    summary="Get explanation history for a patient",
    description="Retrieve past explanations for audit or review purposes."
)
async def get_explanation_history(
    patient_id: str,
    model_type: Optional[ModelType] = None,
    limit: int = 10,
    service: RealTimeSHAPService = Depends(get_shap_service)
) -> Dict[str, Any]:
    """
    Retrieve historical explanations for a patient.
    
    Useful for:
    - Audit trail review
    - Understanding prediction evolution over time
    - Debugging model behavior
    """
    # In production: query ml_shap_analyses table
    return {
        "patient_id": patient_id,
        "total_explanations": 0,
        "explanations": [],
        "message": "Query ml_shap_analyses table for historical data"
    }


@router.get(
    "/models",
    summary="List available models for explanation",
    description="Returns models that support real-time SHAP explanations."
)
async def list_explainable_models() -> Dict[str, Any]:
    """
    List models with SHAP explainer support.
    """
    return {
        "supported_models": [
            {
                "model_type": "propensity",
                "latest_version": "v2.3.1-prod",
                "explainer_type": "TreeExplainer",
                "avg_latency_ms": 85
            },
            {
                "model_type": "risk_stratification",
                "latest_version": "v1.8.0-prod",
                "explainer_type": "TreeExplainer",
                "avg_latency_ms": 92
            },
            {
                "model_type": "next_best_action",
                "latest_version": "v3.1.2-prod",
                "explainer_type": "KernelExplainer",
                "avg_latency_ms": 450
            },
            {
                "model_type": "churn_prediction",
                "latest_version": "v2.0.0-prod",
                "explainer_type": "TreeExplainer",
                "avg_latency_ms": 78
            }
        ],
        "total_models": 4
    }


@router.get(
    "/health",
    summary="Health check for interpretability service"
)
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint for the interpretability service.
    """
    return {
        "status": "healthy",
        "service": "real-time-shap-api",
        "version": "4.1.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dependencies": {
            "bentoml": "connected",
            "feast": "connected",
            "shap_explainer": "loaded",
            "ml_shap_analyses_db": "connected"
        }
    }
