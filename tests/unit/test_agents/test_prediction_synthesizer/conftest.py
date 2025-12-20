"""
E2I Prediction Synthesizer Agent - Test Fixtures
"""

import pytest
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock


# ============================================================================
# MOCK MODEL REGISTRY
# ============================================================================


class MockModelRegistry:
    """Mock model registry for testing."""

    def __init__(self, models: Optional[Dict[str, Dict[str, Any]]] = None):
        self.models = models or {}

    async def get_models_for_target(
        self, target: str, entity_type: str
    ) -> List[str]:
        """Get models that can predict the target."""
        matching = []
        for model_id, info in self.models.items():
            if info.get("target") == target:
                matching.append(model_id)
        return matching

    async def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get info for specific model."""
        return self.models.get(model_id)


# ============================================================================
# MOCK MODEL CLIENT
# ============================================================================


class MockModelClient:
    """Mock model client for testing predictions."""

    def __init__(
        self,
        prediction: float = 0.5,
        confidence: float = 0.8,
        latency_ms: int = 100,
        should_fail: bool = False,
        error_message: str = "Mock error",
    ):
        self.prediction = prediction
        self.confidence = confidence
        self.latency_ms = latency_ms
        self.should_fail = should_fail
        self.error_message = error_message
        self.call_count = 0

    async def predict(
        self,
        entity_id: str,
        features: Dict[str, Any],
        time_horizon: str = "30d",
        **kwargs,
    ) -> Dict[str, Any]:
        """Mock prediction call."""
        self.call_count += 1
        if self.should_fail:
            raise RuntimeError(self.error_message)
        return {
            "prediction": self.prediction,
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
        }


# ============================================================================
# MOCK CONTEXT STORE
# ============================================================================


class MockContextStore:
    """Mock context store for testing."""

    def __init__(
        self,
        similar_cases: Optional[List[Dict[str, Any]]] = None,
        accuracy: float = 0.85,
        history: Optional[List[Dict[str, Any]]] = None,
    ):
        self.similar_cases = similar_cases or []
        self.accuracy = accuracy
        self.history = history or []

    async def find_similar(
        self, entity_type: str, features: Dict[str, Any], limit: int
    ) -> List[Dict[str, Any]]:
        """Find similar historical cases."""
        return self.similar_cases[:limit]

    async def get_accuracy(
        self, prediction_target: str, entity_type: str
    ) -> float:
        """Get historical accuracy."""
        return self.accuracy

    async def get_prediction_history(
        self, entity_id: str, prediction_target: str, limit: int
    ) -> List[Dict[str, Any]]:
        """Get prediction history."""
        return self.history[:limit]


# ============================================================================
# MOCK FEATURE STORE
# ============================================================================


class MockFeatureStore:
    """Mock feature store for testing."""

    def __init__(
        self,
        importance: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        self.importance = importance or {}

    async def get_importance(self, model_id: str) -> Dict[str, float]:
        """Get feature importance for model."""
        return self.importance.get(model_id, {})


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_model_registry():
    """Create mock model registry with test models."""
    return MockModelRegistry(
        models={
            "churn_xgb": {
                "target": "churn",
                "entity_type": "hcp",
                "version": "1.0",
            },
            "churn_rf": {
                "target": "churn",
                "entity_type": "hcp",
                "version": "1.0",
            },
            "churn_nn": {
                "target": "churn",
                "entity_type": "hcp",
                "version": "1.0",
            },
            "conversion_xgb": {
                "target": "conversion",
                "entity_type": "hcp",
                "version": "1.0",
            },
        }
    )


@pytest.fixture
def mock_model_clients():
    """Create mock model clients."""
    return {
        "churn_xgb": MockModelClient(prediction=0.72, confidence=0.88, latency_ms=50),
        "churn_rf": MockModelClient(prediction=0.68, confidence=0.82, latency_ms=60),
        "churn_nn": MockModelClient(prediction=0.75, confidence=0.85, latency_ms=80),
        "conversion_xgb": MockModelClient(prediction=0.45, confidence=0.90, latency_ms=55),
    }


@pytest.fixture
def failing_model_clients():
    """Create model clients where some fail."""
    return {
        "churn_xgb": MockModelClient(prediction=0.72, confidence=0.88),
        "churn_rf": MockModelClient(should_fail=True, error_message="Model timeout"),
        "churn_nn": MockModelClient(prediction=0.75, confidence=0.85),
    }


@pytest.fixture
def mock_context_store():
    """Create mock context store."""
    return MockContextStore(
        similar_cases=[
            {"entity_id": "hcp_100", "prediction": 0.65, "outcome": 1},
            {"entity_id": "hcp_101", "prediction": 0.70, "outcome": 1},
            {"entity_id": "hcp_102", "prediction": 0.60, "outcome": 0},
        ],
        accuracy=0.82,
        history=[
            {"prediction": 0.40, "timestamp": "2024-01-01"},
            {"prediction": 0.50, "timestamp": "2024-02-01"},
            {"prediction": 0.60, "timestamp": "2024-03-01"},
            {"prediction": 0.70, "timestamp": "2024-04-01"},
            {"prediction": 0.80, "timestamp": "2024-05-01"},
        ],
    )


@pytest.fixture
def mock_feature_store():
    """Create mock feature store."""
    return MockFeatureStore(
        importance={
            "churn_xgb": {
                "call_frequency": 0.25,
                "prescription_count": 0.20,
                "territory_market_share": 0.15,
                "engagement_score": 0.12,
            },
            "churn_rf": {
                "call_frequency": 0.22,
                "prescription_count": 0.18,
                "territory_market_share": 0.16,
                "days_since_last_call": 0.14,
            },
            "churn_nn": {
                "call_frequency": 0.28,
                "prescription_count": 0.22,
                "engagement_score": 0.15,
            },
        }
    )


@pytest.fixture
def sample_features():
    """Sample feature values for prediction."""
    return {
        "call_frequency": 12,
        "prescription_count": 45,
        "territory_market_share": 0.15,
        "engagement_score": 0.72,
        "days_since_last_call": 14,
    }


@pytest.fixture
def base_state(sample_features):
    """Create base state for testing."""
    return {
        "query": "What is the churn risk for HCP-123?",
        "entity_id": "hcp_123",
        "entity_type": "hcp",
        "prediction_target": "churn",
        "features": sample_features,
        "time_horizon": "30d",
        "models_to_use": None,
        "ensemble_method": "weighted",
        "confidence_level": 0.95,
        "include_context": True,
        "individual_predictions": None,
        "models_succeeded": 0,
        "models_failed": 0,
        "ensemble_prediction": None,
        "prediction_summary": None,
        "prediction_context": None,
        "orchestration_latency_ms": 0,
        "ensemble_latency_ms": 0,
        "total_latency_ms": 0,
        "timestamp": "",
        "errors": [],
        "warnings": [],
        "status": "pending",
    }


@pytest.fixture
def state_with_predictions(base_state):
    """State after orchestration with predictions."""
    return {
        **base_state,
        "individual_predictions": [
            {
                "model_id": "churn_xgb",
                "prediction": 0.72,
                "confidence": 0.88,
                "latency_ms": 50,
            },
            {
                "model_id": "churn_rf",
                "prediction": 0.68,
                "confidence": 0.82,
                "latency_ms": 60,
            },
        ],
        "models_succeeded": 2,
        "models_failed": 0,
        "orchestration_latency_ms": 65,
        "status": "combining",
    }


@pytest.fixture
def state_with_ensemble(state_with_predictions):
    """State after ensemble combination."""
    return {
        **state_with_predictions,
        "ensemble_prediction": {
            "point_estimate": 0.70,
            "prediction_interval_lower": 0.58,
            "prediction_interval_upper": 0.82,
            "confidence": 0.85,
            "ensemble_method": "weighted",
            "model_agreement": 0.95,
        },
        "prediction_summary": "Prediction: 0.700 (95% CI: [0.580, 0.820]). Confidence: high. Model agreement: strong across 2 models.",
        "ensemble_latency_ms": 5,
        "status": "enriching",
    }
