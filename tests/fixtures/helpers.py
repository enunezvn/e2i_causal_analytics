"""Shared helper functions for fixture creation.

Provides reusable functions for creating common test data structures
used across multiple test modules.

Usage:
    from tests.fixtures.helpers import make_decomposition_response

    response = make_decomposition_response(
        sub_questions=["What is TRx?", "What is market share?"]
    )
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4


# =============================================================================
# TOOL COMPOSER HELPERS
# =============================================================================


def make_decomposition_response(
    sub_questions: Optional[List[str]] = None,
    reasoning: str = "Query decomposed into sub-questions",
) -> str:
    """Create a decomposition response for tool composer tests.

    Args:
        sub_questions: List of sub-questions (default: 2 sample questions)
        reasoning: Explanation of decomposition logic

    Returns:
        JSON string with decomposition structure
    """
    if sub_questions is None:
        sub_questions = [
            "What is the current TRx value?",
            "What is the trend over time?",
        ]

    return json.dumps({
        "sub_questions": [
            {"id": f"q{i+1}", "question": q, "dependencies": []}
            for i, q in enumerate(sub_questions)
        ],
        "reasoning": reasoning,
        "is_multi_faceted": len(sub_questions) > 1,
    })


def make_planning_response(
    steps: Optional[List[Dict[str, Any]]] = None,
    strategy: str = "sequential",
) -> str:
    """Create a planning response for tool composer tests.

    Args:
        steps: List of execution steps (default: 2 sample steps)
        strategy: Execution strategy ("sequential" or "parallel")

    Returns:
        JSON string with execution plan structure
    """
    if steps is None:
        steps = [
            {"step": 1, "tool": "kpi_query", "question_id": "q1", "params": {}},
            {"step": 2, "tool": "trend_analysis", "question_id": "q2", "params": {}},
        ]

    return json.dumps({
        "execution_plan": steps,
        "strategy": strategy,
        "estimated_calls": len(steps),
    })


def make_synthesis_response(
    answer: str = "Synthesized answer based on sub-question results.",
    confidence: float = 0.85,
    sources: Optional[List[str]] = None,
) -> str:
    """Create a synthesis response for tool composer tests.

    Args:
        answer: The final synthesized answer
        confidence: Confidence score (0-1)
        sources: List of source references

    Returns:
        JSON string with synthesis structure
    """
    return json.dumps({
        "final_answer": answer,
        "confidence": confidence,
        "sources": sources or ["q1_result", "q2_result"],
        "synthesis_method": "aggregate",
    })


# =============================================================================
# EXPERIMENT MONITOR HELPERS
# =============================================================================


def create_experiment_summary(
    experiment_id: Optional[str] = None,
    name: str = "Test Experiment",
    status: str = "running",
    start_date: Optional[datetime] = None,
    metrics: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Create an experiment summary for monitor tests.

    Args:
        experiment_id: Unique experiment ID (generated if not provided)
        name: Experiment name
        status: Current status (running, completed, paused, failed)
        start_date: Experiment start date
        metrics: Current metric values

    Returns:
        Experiment summary dictionary
    """
    return {
        "experiment_id": experiment_id or str(uuid4()),
        "name": name,
        "status": status,
        "start_date": (start_date or datetime.utcnow()).isoformat(),
        "metrics": metrics or {
            "conversion_rate": 0.05,
            "sample_size": 1000,
            "p_value": 0.03,
        },
        "variants": ["control", "treatment"],
    }


def create_srm_issue(
    experiment_id: str,
    expected_ratio: float = 0.5,
    observed_ratio: float = 0.55,
    severity: str = "warning",
) -> Dict[str, Any]:
    """Create a Sample Ratio Mismatch (SRM) issue.

    Args:
        experiment_id: Associated experiment ID
        expected_ratio: Expected sample ratio
        observed_ratio: Observed sample ratio
        severity: Issue severity (warning, error, critical)

    Returns:
        SRM issue dictionary
    """
    return {
        "issue_type": "srm",
        "experiment_id": experiment_id,
        "expected_ratio": expected_ratio,
        "observed_ratio": observed_ratio,
        "deviation": abs(observed_ratio - expected_ratio),
        "severity": severity,
        "detected_at": datetime.utcnow().isoformat(),
        "recommendation": "Investigate assignment mechanism for bias",
    }


def create_enrollment_issue(
    experiment_id: str,
    target_size: int = 10000,
    current_size: int = 5000,
    days_remaining: int = 7,
) -> Dict[str, Any]:
    """Create an enrollment issue for experiment monitoring.

    Args:
        experiment_id: Associated experiment ID
        target_size: Target sample size
        current_size: Current enrolled sample size
        days_remaining: Days until experiment deadline

    Returns:
        Enrollment issue dictionary
    """
    return {
        "issue_type": "enrollment",
        "experiment_id": experiment_id,
        "target_size": target_size,
        "current_size": current_size,
        "progress_pct": (current_size / target_size) * 100,
        "days_remaining": days_remaining,
        "projected_completion": current_size / max(1, (target_size - current_size) / days_remaining),
        "detected_at": datetime.utcnow().isoformat(),
        "recommendation": "Consider extending experiment duration or increasing traffic",
    }


def create_alert(
    alert_type: str = "experiment_issue",
    severity: str = "warning",
    message: str = "Test alert message",
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create an alert for monitoring tests.

    Args:
        alert_type: Type of alert
        severity: Alert severity (info, warning, error, critical)
        message: Alert message
        context: Additional context data

    Returns:
        Alert dictionary
    """
    return {
        "alert_id": str(uuid4()),
        "alert_type": alert_type,
        "severity": severity,
        "message": message,
        "context": context or {},
        "created_at": datetime.utcnow().isoformat(),
        "acknowledged": False,
    }


# =============================================================================
# CAUSAL ANALYSIS HELPERS
# =============================================================================


def create_causal_path(
    source: str = "marketing_spend",
    target: str = "TRx",
    effect_size: float = 0.15,
    confidence: float = 0.95,
    mechanism: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a causal path for analysis tests.

    Args:
        source: Source variable
        target: Target variable
        effect_size: Estimated causal effect
        confidence: Confidence in the causal relationship
        mechanism: Optional explanation of the mechanism

    Returns:
        Causal path dictionary
    """
    return {
        "source": source,
        "target": target,
        "effect_size": effect_size,
        "confidence": confidence,
        "mechanism": mechanism or f"{source} influences {target} through market dynamics",
        "discovered_at": datetime.utcnow().isoformat(),
    }


def create_gap_analysis_result(
    gap_type: str = "market_opportunity",
    current_value: float = 0.25,
    target_value: float = 0.35,
    roi_potential: float = 1.5,
) -> Dict[str, Any]:
    """Create a gap analysis result.

    Args:
        gap_type: Type of gap identified
        current_value: Current metric value
        target_value: Target metric value
        roi_potential: Potential ROI if gap is closed

    Returns:
        Gap analysis result dictionary
    """
    return {
        "gap_type": gap_type,
        "current_value": current_value,
        "target_value": target_value,
        "gap_size": target_value - current_value,
        "roi_potential": roi_potential,
        "priority_score": roi_potential * (target_value - current_value),
        "recommendations": [
            "Increase marketing investment in underperforming regions",
            "Target high-potential HCPs with personalized outreach",
        ],
    }


# =============================================================================
# DRIFT MONITORING HELPERS
# =============================================================================


def create_drift_report(
    drift_type: str = "data_drift",
    feature: str = "patient_age",
    baseline_mean: float = 55.0,
    current_mean: float = 58.0,
    p_value: float = 0.01,
    is_significant: bool = True,
) -> Dict[str, Any]:
    """Create a drift detection report.

    Args:
        drift_type: Type of drift (data_drift, concept_drift, label_drift)
        feature: Feature with drift
        baseline_mean: Baseline period mean
        current_mean: Current period mean
        p_value: Statistical significance
        is_significant: Whether drift is significant

    Returns:
        Drift report dictionary
    """
    return {
        "drift_type": drift_type,
        "feature": feature,
        "baseline_stats": {
            "mean": baseline_mean,
            "std": 10.0,
            "n_samples": 1000,
        },
        "current_stats": {
            "mean": current_mean,
            "std": 12.0,
            "n_samples": 500,
        },
        "p_value": p_value,
        "is_significant": is_significant,
        "detected_at": datetime.utcnow().isoformat(),
        "recommendation": "Review recent data pipeline changes" if is_significant else "Continue monitoring",
    }


# =============================================================================
# PREDICTION HELPERS
# =============================================================================


def create_prediction_result(
    model_id: str = "model_v1",
    prediction: float = 0.75,
    confidence_interval: Optional[tuple] = None,
    features_used: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Create a prediction result.

    Args:
        model_id: Model identifier
        prediction: Predicted value
        confidence_interval: Optional (lower, upper) bounds
        features_used: List of features used in prediction

    Returns:
        Prediction result dictionary
    """
    return {
        "model_id": model_id,
        "prediction": prediction,
        "confidence_interval": confidence_interval or (0.70, 0.80),
        "features_used": features_used or ["feature_a", "feature_b", "feature_c"],
        "predicted_at": datetime.utcnow().isoformat(),
        "metadata": {
            "model_version": "1.0.0",
            "inference_time_ms": 15,
        },
    }


def create_ensemble_prediction(
    predictions: Optional[List[Dict[str, Any]]] = None,
    aggregation_method: str = "weighted_average",
) -> Dict[str, Any]:
    """Create an ensemble prediction result.

    Args:
        predictions: Individual model predictions
        aggregation_method: How predictions were combined

    Returns:
        Ensemble prediction dictionary
    """
    if predictions is None:
        predictions = [
            create_prediction_result(model_id="model_1", prediction=0.72),
            create_prediction_result(model_id="model_2", prediction=0.78),
            create_prediction_result(model_id="model_3", prediction=0.75),
        ]

    values = [p["prediction"] for p in predictions]
    ensemble_value = sum(values) / len(values)

    return {
        "ensemble_prediction": ensemble_value,
        "individual_predictions": predictions,
        "aggregation_method": aggregation_method,
        "model_count": len(predictions),
        "prediction_variance": max(values) - min(values),
        "created_at": datetime.utcnow().isoformat(),
    }


# =============================================================================
# FEEDBACK HELPERS
# =============================================================================


def create_feedback_item(
    rating: int = 4,
    comment: Optional[str] = None,
    query: str = "What is the TRx trend?",
    response_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a feedback item for feedback learner tests.

    Args:
        rating: User rating (1-5)
        comment: Optional user comment
        query: Original query
        response_id: ID of the response being rated

    Returns:
        Feedback item dictionary
    """
    return {
        "feedback_id": str(uuid4()),
        "response_id": response_id or str(uuid4()),
        "query": query,
        "rating": rating,
        "comment": comment,
        "feedback_type": "explicit",
        "created_at": datetime.utcnow().isoformat(),
        "user_id": "test-user",
    }


def create_feedback_pattern(
    pattern_type: str = "query_complexity",
    description: str = "Complex multi-part queries tend to receive lower ratings",
    confidence: float = 0.85,
    sample_size: int = 50,
) -> Dict[str, Any]:
    """Create a detected feedback pattern.

    Args:
        pattern_type: Type of pattern detected
        description: Pattern description
        confidence: Confidence in pattern detection
        sample_size: Number of feedback items analyzed

    Returns:
        Pattern dictionary
    """
    return {
        "pattern_id": str(uuid4()),
        "pattern_type": pattern_type,
        "description": description,
        "confidence": confidence,
        "sample_size": sample_size,
        "detected_at": datetime.utcnow().isoformat(),
        "actionable": True,
        "suggested_action": "Break complex queries into simpler sub-questions",
    }
