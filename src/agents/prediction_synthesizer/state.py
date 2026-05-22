"""
E2I Prediction Synthesizer Agent - State Definitions
Version: 4.2
Purpose: LangGraph state for prediction aggregation and ensemble
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict
from uuid import UUID

from typing_extensions import NotRequired


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
    """Context for interpreting prediction.

    Per-field availability flags (added for #438 F-012-followup) disambiguate
    real-but-empty from dependency-failure outcomes:

    - `*_available=True` + populated value -> dependency returned data
    - `*_available=True` + empty value -> dependency returned honest-empty
      (e.g., zero history, no store configured)
    - `*_available=False` + sentinel value -> dependency raised Exception;
      a per-field warning naming the failed dependency is appended to
      top-level `warnings`.

    Sentinel-on-failure values are non-plausible (None for historical_accuracy
    and trend_direction; [] / {} for list/dict-shaped fields) so downstream
    consumers cannot confuse "no signal" with "real zero" / "real stable".

    All availability flags are `NotRequired` for back-compat with existing
    PredictionContext construction sites; absent flag defaults are treated
    as legacy True (best-effort) by consumers that don't inspect them.
    """

    similar_cases: List[Dict[str, Any]]
    feature_importance: Dict[str, float]
    # Optional[float]/Optional[Literal...] (None sentinel on dep failure)
    historical_accuracy: NotRequired[Optional[float]]
    trend_direction: NotRequired[Optional[Literal["increasing", "stable", "decreasing"]]]
    # Per-field availability flags (added 2026-05-22 for #438)
    similar_cases_available: NotRequired[bool]
    feature_importance_available: NotRequired[bool]
    historical_accuracy_available: NotRequired[bool]
    trend_direction_available: NotRequired[bool]
    online_features_available: NotRequired[bool]


class PredictionSynthesizerState(TypedDict):
    """Complete state for Prediction Synthesizer agent"""

    # === INPUT (optional on initialization) ===
    query: NotRequired[str]
    entity_id: NotRequired[str]  # HCP ID, territory ID, etc.
    entity_type: NotRequired[str]  # "hcp", "territory", "patient"
    prediction_target: NotRequired[str]  # What to predict
    features: NotRequired[Dict[str, Any]]
    time_horizon: NotRequired[str]  # e.g., "30d", "90d"

    # === CONFIGURATION (optional) ===
    models_to_use: NotRequired[Optional[List[str]]]  # Specific models, or None for all
    ensemble_method: NotRequired[Literal["average", "weighted", "stacking", "voting"]]
    confidence_level: NotRequired[float]  # Default: 0.95
    include_context: NotRequired[bool]

    # === MODEL OUTPUTS ===
    individual_predictions: NotRequired[Optional[List[ModelPrediction]]]
    models_succeeded: NotRequired[int]
    models_failed: NotRequired[int]

    # === ENSEMBLE OUTPUTS ===
    ensemble_prediction: NotRequired[Optional[EnsemblePrediction]]
    prediction_summary: NotRequired[Optional[str]]

    # === CONTEXT OUTPUTS ===
    prediction_context: NotRequired[Optional[PredictionContext]]

    # === INTERPRETATION (P2 enhancement) ===
    prediction_interpretation: NotRequired[
        Optional[Dict[str, Any]]
    ]  # Risk assessment, anomalies, recommendations

    # === EXECUTION METADATA ===
    orchestration_latency_ms: NotRequired[int]
    ensemble_latency_ms: NotRequired[int]
    total_latency_ms: NotRequired[int]
    timestamp: str  # Required output

    # === ERROR HANDLING ===
    # Note: Required outputs for contract compliance
    errors: Annotated[List[Dict[str, Any]], operator.add]
    warnings: Annotated[List[str], operator.add]
    status: Literal[
        "pending",
        "predicting",
        "combining",
        "enriching",
        "completed",
        # "degraded" added for #438: 3-5 of 5 context_enricher deps failed but
        # enrichment surface remains non-fatal-but-honest. Distinct from "failed"
        # (catastrophic / contract broken) and "completed" (full success or
        # 0-2 dep failures with availability flags surfaced honestly).
        "degraded",
        "failed",
    ]

    # === AUDIT CHAIN ===
    audit_workflow_id: NotRequired[UUID]
