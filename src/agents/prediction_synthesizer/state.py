"""
E2I Prediction Synthesizer Agent - State Definitions
Version: 4.2
Purpose: LangGraph state for prediction aggregation and ensemble
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict
from uuid import UUID


class ModelPrediction(TypedDict):
    """Individual model prediction"""

    model_id: str
    model_type: str
    prediction: float
    prediction_proba: Optional[List[float]]
    confidence: float
    latency_ms: int
    features_used: List[str]


class EnsemblePrediction(TypedDict):
    """Combined ensemble prediction"""

    point_estimate: float
    prediction_interval_lower: float
    prediction_interval_upper: float
    confidence: float
    ensemble_method: str
    model_agreement: float  # 0-1, how much models agree


class PredictionContext(TypedDict):
    """Context for interpreting prediction"""

    similar_cases: List[Dict[str, Any]]
    feature_importance: Dict[str, float]
    historical_accuracy: float
    trend_direction: Literal["increasing", "stable", "decreasing"]


class PredictionSynthesizerState(TypedDict, total=False):
    """Complete state for Prediction Synthesizer agent"""

    # === INPUT ===
    query: str
    entity_id: str  # HCP ID, territory ID, etc.
    entity_type: str  # "hcp", "territory", "patient"
    prediction_target: str  # What to predict
    features: Dict[str, Any]
    time_horizon: str  # e.g., "30d", "90d"

    # === CONFIGURATION ===
    models_to_use: Optional[List[str]]  # Specific models, or None for all
    ensemble_method: Literal["average", "weighted", "stacking", "voting"]
    confidence_level: float  # Default: 0.95
    include_context: bool

    # === MODEL OUTPUTS ===
    individual_predictions: Optional[List[ModelPrediction]]
    models_succeeded: int
    models_failed: int

    # === ENSEMBLE OUTPUTS ===
    ensemble_prediction: Optional[EnsemblePrediction]
    prediction_summary: Optional[str]

    # === CONTEXT OUTPUTS ===
    prediction_context: Optional[PredictionContext]

    # === EXECUTION METADATA ===
    orchestration_latency_ms: int
    ensemble_latency_ms: int
    total_latency_ms: int
    timestamp: str

    # === ERROR HANDLING ===
    errors: Annotated[List[Dict[str, Any]], operator.add]
    warnings: Annotated[List[str], operator.add]
    status: Literal["pending", "predicting", "combining", "enriching", "completed", "failed"]

    # === AUDIT CHAIN ===
    audit_workflow_id: UUID
