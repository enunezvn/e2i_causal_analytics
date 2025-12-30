"""Feedback Loop Test Fixtures.

Reusable fixtures for feedback loop integration testing.
Memory-safe with small batch sizes per CLAUDE.md requirements.

Author: E2I Causal Analytics Team
Version: 1.0.0
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
from uuid import uuid4

import pytest


# =============================================================================
# PREDICTION FIXTURES
# =============================================================================


@pytest.fixture
def small_prediction_batch() -> List[Dict[str, Any]]:
    """Generate a small batch of 10 predictions for testing.

    Memory-safe for low-resource environments.
    Covers multiple prediction types for comprehensive testing.
    """
    base_time = datetime.now(timezone.utc) - timedelta(days=30)
    prediction_types = [
        "trigger",
        "next_best_action",
        "hcp_churn",
        "market_share_impact",
        "risk",
    ]

    predictions = []
    for i in range(10):
        pred_type = prediction_types[i % len(prediction_types)]
        predictions.append({
            "prediction_id": str(uuid4()),
            "prediction_type": pred_type,
            "entity_type": "hcp",
            "entity_id": f"hcp_{i:03d}",
            "prediction_value": 0.3 + (i * 0.05),  # Range: 0.3 to 0.75
            "confidence": 0.7 + (i * 0.02),  # Range: 0.70 to 0.88
            "created_at": (base_time + timedelta(days=i)).isoformat(),
            "brand": ["remibrutinib", "fabhalta", "kisqali"][i % 3],
            "model_version": "v1.0.0",
            "features_used": ["feature_a", "feature_b", "feature_c"],
        })

    return predictions


@pytest.fixture
def labeled_prediction_batch() -> List[Dict[str, Any]]:
    """Generate predictions with ground truth labels assigned.

    For testing concept drift detection after feedback loop.
    """
    base_time = datetime.now(timezone.utc) - timedelta(days=60)
    outcome_time = datetime.now(timezone.utc) - timedelta(days=10)

    predictions = []
    for i in range(10):
        # Alternate between correct and incorrect predictions
        predicted_positive = i % 2 == 0
        actual_positive = i % 3 != 0  # Different pattern for drift

        predictions.append({
            "prediction_id": str(uuid4()),
            "prediction_type": "trigger",
            "entity_type": "hcp",
            "entity_id": f"hcp_{i:03d}",
            "prediction_value": 0.75 if predicted_positive else 0.25,
            "confidence": 0.85,
            "created_at": (base_time + timedelta(days=i)).isoformat(),
            "brand": "remibrutinib",
            # Ground truth labels
            "actual_outcome": 1 if actual_positive else 0,
            "outcome_recorded_at": (outcome_time + timedelta(days=i)).isoformat(),
            "truth_confidence": 0.95,
            "truth_source": "crm_activity",
        })

    return predictions


# =============================================================================
# FEEDBACK LOOP RESULT FIXTURES
# =============================================================================


@pytest.fixture
def mock_feedback_loop_result() -> Dict[str, Any]:
    """Mock response from PL/pgSQL run_feedback_loop() function.

    Simulates a successful feedback loop execution.
    """
    return {
        "run_id": str(uuid4()),
        "predictions_labeled": 25,
        "predictions_skipped": 5,
        "status": "completed",
        "prediction_type": "trigger",
        "execution_time_ms": 1250,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@pytest.fixture
def mock_feedback_loop_result_empty() -> Dict[str, Any]:
    """Mock response when no predictions to label."""
    return {
        "run_id": str(uuid4()),
        "predictions_labeled": 0,
        "predictions_skipped": 0,
        "status": "completed",
        "prediction_type": "trigger",
        "execution_time_ms": 50,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@pytest.fixture
def mock_feedback_loop_partial_result() -> Dict[str, Any]:
    """Mock response for partial success (some predictions had errors)."""
    return {
        "run_id": str(uuid4()),
        "predictions_labeled": 15,
        "predictions_skipped": 10,
        "predictions_error": 5,
        "status": "partial",
        "prediction_type": "hcp_churn",
        "execution_time_ms": 2500,
        "errors": [
            {"entity_id": "hcp_001", "error": "Missing outcome data"},
            {"entity_id": "hcp_005", "error": "Invalid entity reference"},
        ],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# =============================================================================
# DRIFT ALERT FIXTURES
# =============================================================================


@pytest.fixture
def mock_drift_alert_data() -> List[Dict[str, Any]]:
    """Mock data from v_drift_alerts view.

    Simulates drift detection results after feedback loop processing.
    """
    return [
        {
            "prediction_type": "trigger",
            "accuracy_status": "OK",
            "calibration_status": "OK",
            "accuracy_drop": 0.02,
            "calibration_error": 0.03,
            "baseline_accuracy": 0.82,
            "current_accuracy": 0.80,
            "predictions_count": 500,
            "alert_timestamp": datetime.now(timezone.utc).isoformat(),
        },
        {
            "prediction_type": "hcp_churn",
            "accuracy_status": "ALERT",
            "calibration_status": "WARNING",
            "accuracy_drop": 0.08,
            "calibration_error": 0.12,
            "baseline_accuracy": 0.78,
            "current_accuracy": 0.70,
            "predictions_count": 250,
            "alert_timestamp": datetime.now(timezone.utc).isoformat(),
        },
    ]


@pytest.fixture
def mock_drift_alert_critical() -> Dict[str, Any]:
    """Mock critical drift alert for testing alert routing."""
    return {
        "prediction_type": "market_share_impact",
        "accuracy_status": "ALERT",
        "calibration_status": "ALERT",
        "accuracy_drop": 0.15,  # >10% = critical
        "calibration_error": 0.18,
        "baseline_accuracy": 0.85,
        "current_accuracy": 0.70,
        "predictions_count": 100,
        "class_ratio_shift": 0.20,
        "alert_timestamp": datetime.now(timezone.utc).isoformat(),
    }


@pytest.fixture
def mock_concept_drift_metrics() -> List[Dict[str, Any]]:
    """Mock data from v_concept_drift_metrics view."""
    base_date = datetime.now(timezone.utc) - timedelta(weeks=4)
    return [
        {
            "week_start": (base_date + timedelta(weeks=i)).isoformat(),
            "prediction_type": "trigger",
            "total_predictions": 100 + (i * 10),
            "labeled_predictions": 90 + (i * 8),
            "accuracy": 0.82 - (i * 0.01),  # Declining accuracy
            "calibration_error": 0.03 + (i * 0.005),
            "true_positive_rate": 0.78 - (i * 0.02),
            "false_positive_rate": 0.12 + (i * 0.01),
        }
        for i in range(4)
    ]


# =============================================================================
# SUPABASE CLIENT MOCK FIXTURES
# =============================================================================


@pytest.fixture
def mock_supabase_rpc_response():
    """Factory fixture for creating mock Supabase RPC responses."""
    def _create_response(data: Any, error: Any = None):
        class MockResponse:
            def __init__(self, data, error):
                self.data = data
                self.error = error

        return MockResponse(data, error)
    return _create_response


@pytest.fixture
def mock_supabase_table_response():
    """Factory fixture for creating mock Supabase table query responses."""
    def _create_response(data: List[Dict], count: int = None):
        class MockQueryResponse:
            def __init__(self, data, count):
                self.data = data
                self.count = count or len(data)

        return MockQueryResponse(data, count)
    return _create_response


# =============================================================================
# CELERY TASK FIXTURES
# =============================================================================


@pytest.fixture
def mock_celery_request():
    """Mock Celery task request object."""
    class MockRequest:
        id = str(uuid4())
        retries = 0
        delivery_info = {"routing_key": "analytics"}
        hostname = "test-worker@localhost"

    return MockRequest()


@pytest.fixture
def mock_celery_task_result():
    """Mock Celery AsyncResult for chained tasks."""
    class MockAsyncResult:
        def __init__(self, task_id: str = None):
            self.id = task_id or str(uuid4())
            self.status = "PENDING"
            self.result = None

        def get(self, timeout: float = None):
            return self.result

        def ready(self):
            return self.status in ("SUCCESS", "FAILURE")

    return MockAsyncResult


# =============================================================================
# CONFIGURATION FIXTURES
# =============================================================================


@pytest.fixture
def mock_feedback_loop_config() -> Dict[str, Any]:
    """Mock configuration for feedback loop tasks."""
    return {
        "feedback_loop": {
            "schedule": {
                "short_window_types": ["trigger", "next_best_action"],
                "medium_window_types": ["hcp_churn"],
                "long_window_types": ["market_share_impact", "risk"],
            },
            "processing": {
                "batch_size": 1000,
                "max_retries": 3,
                "retry_delay_minutes": 15,
                "min_confidence_threshold": 0.60,
            },
            "alerts": {
                "accuracy_degradation_threshold": 0.10,
                "indeterminate_rate_threshold": 0.20,
            },
        },
        "drift_integration": {
            "concept_drift": {
                "enabled": True,
                "comparison_windows": {
                    "baseline_days": 90,
                    "current_days": 30,
                },
                "alert_thresholds": {
                    "accuracy_drop": 0.05,
                    "calibration_error": 0.10,
                    "class_shift": 0.15,
                },
            },
        },
    }


@pytest.fixture
def mock_outcome_truth_rules() -> Dict[str, Any]:
    """Mock outcome truth rules configuration."""
    return {
        "prediction_types": {
            "trigger": {
                "observation_window_days": 14,
                "truth_sources": ["crm_activity", "call_logs"],
                "positive_signals": ["meeting_scheduled", "call_completed"],
                "confidence_threshold": 0.70,
            },
            "hcp_churn": {
                "observation_window_days": 90,
                "truth_sources": ["rx_data", "engagement_scores"],
                "positive_signals": ["rx_decrease_50pct", "engagement_drop"],
                "confidence_threshold": 0.75,
            },
        },
    }
